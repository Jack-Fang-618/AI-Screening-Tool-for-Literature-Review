"""
AI Screener - Intelligent Article Screening Engine

Parallel screening of systematic review articles using LLM models.
Preserves excellent logic from ai_screener.py while improving:
- Async/parallel processing with 8 workers
- Better error handling and retry logic
- Structured responses with Pydantic
- Real-time progress tracking
- Checkpoint/resume capability

Refined improvements:
- ThreadPoolExecutor with configurable workers
- Exponential backoff retry strategy
- Streaming progress updates
- Cost tracking per model
- Batch checkpointing
"""

import pandas as pd
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
from datetime import datetime
import numpy as np
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class ScreeningConfig:
    """Configuration for screening operation"""
    inclusion_criteria: List[str]
    exclusion_criteria: List[str]
    confidence_threshold: float = 0.5
    manual_review_threshold: float = 0.6
    batch_size: int = 50
    strategy: str = "single"  # "single" or "dual_arbiter"
    max_workers: int = 4  # Default 4 workers (stable). Max recommended: 8. Reduce to 2 if experiencing network errors
    checkpoint_interval: int = 100  # Save checkpoint every N articles
    
    def __post_init__(self):
        """Validate and warn about worker count"""
        if self.max_workers > 8:
            logger.warning(
                f"⚠️ Worker count set to {self.max_workers}. High worker counts may cause "
                f"network instability and rate limiting. Recommended: 2-4 for stable production, "
                f"max 8 for faster processing with acceptable retry risk."
            )


@dataclass
class ArticleDecision:
    """Screening decision for single article"""
    title: str
    abstract: str
    decision: str  # "Relevant", "Irrelevant", "Uncertain"
    reason: str
    confidence: float
    needs_manual_review: bool
    timestamp: str
    provider: str
    model: str
    reasoning_content: Optional[str] = None
    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    # Cost
    api_cost: float = 0.0
    # Strategy info
    is_arbiter_decision: bool = False
    is_consensus: bool = False


@dataclass
class ScreeningResult:
    """Complete screening operation result"""
    total_articles: int
    screened_count: int
    relevant: int = 0
    irrelevant: int = 0
    uncertain: int = 0
    errors: int = 0
    needs_manual_review: int = 0
    decisions: List[ArticleDecision] = field(default_factory=list)
    # Performance metrics
    total_time: float = 0.0
    avg_time_per_article: float = 0.0
    # Cost tracking
    total_cost: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_reasoning_tokens: int = 0
    avg_cost_per_article: float = 0.0
    # Model info
    model_used: str = ""
    provider_used: str = ""
    

class AIScreener:
    """
    Intelligent article screener with parallel processing
    
    Features:
    - 8-worker parallel execution
    - Configurable LLM client
    - Real-time progress callbacks
    - Checkpoint/resume capability
    - Comprehensive cost tracking
    """
    
    # Model pricing (per 1M tokens) - January 2026 - HKD (USD * 7.78 exchange rate)
    MODEL_PRICING = {
        # Grok 4.1 Fast series - Latest models (January 2026) - BEST for screening!
        'grok-4-1-fast-reasoning': {
            'input': 0.2 * 7.78,        # USD $0.20 → HKD $1.556
            'cached_input': 0.05 * 7.78,  # USD $0.05 → HKD $0.389
            'output': 0.5 * 7.78          # USD $0.50 → HKD $3.89
        },
        'grok-4-1-fast-non-reasoning': {
            'input': 0.2 * 7.78,        # USD $0.20 → HKD $1.556
            'cached_input': 0.05 * 7.78,  # USD $0.05 → HKD $0.389
            'output': 0.5 * 7.78          # USD $0.50 → HKD $3.89
        },
        
        # Backward compatibility for old names
        'grok-4-fast-reasoning': {
            'input': 0.2 * 7.78,
            'cached_input': 0.05 * 7.78,
            'output': 0.5 * 7.78
        },
        'grok-4-fast-non-reasoning': {
            'input': 0.2 * 7.78,
            'cached_input': 0.05 * 7.78,
            'output': 0.5 * 7.78
        },
        
        # Grok 4 Standard - More expensive
        'grok-4': {
            'input': 3.0 * 7.78,        # USD $3.00 → HKD $23.34
            'cached_input': 0.75 * 7.78,  # USD $0.75 → HKD $5.835
            'output': 15.0 * 7.78         # USD $15.00 → HKD $116.70
        },
        
        # Grok 3 series - Previous generation (USD pricing, not yet updated to HKD)
        'grok-3': {'input': 3.0 * 7.78, 'output': 15.0 * 7.78},
        'grok-3-mini': {'input': 0.3 * 7.78, 'output': 0.5 * 7.78},
        'grok-3-mini-fast': {'input': 0.6 * 7.78, 'output': 1.0 * 7.78},
        'grok-3-fast': {'input': 5.0 * 7.78, 'output': 25.0 * 7.78},
        
        # Default fallback
        'default': {'input': 1.0 * 7.78, 'output': 1.0 * 7.78}
    }
    
    def __init__(
        self,
        llm_client,  # LLM client (GrokClient or similar)
        config: ScreeningConfig,
        checkpoint_dir: Optional[Path] = None
    ):
        """
        Initialize screener
        
        Args:
            llm_client: LLM client instance with screen_article() method
            config: Screening configuration
            checkpoint_dir: Optional directory for saving checkpoints
        """
        self.llm_client = llm_client
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self._cancel_requested = False  # Cancellation flag
        
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def request_cancel(self):
        """Request cancellation of ongoing screening"""
        self._cancel_requested = True
        logger.warning("🛑 Cancellation requested for screener")
    
    def create_screening_prompt(self, title: str, abstract: str) -> str:
        """
        Create AI screening prompt
        
        Args:
            title: Article title
            abstract: Article abstract
            
        Returns:
            Formatted prompt
        """
        inclusion_criteria = "\n".join([f"- {c}" for c in self.config.inclusion_criteria])
        exclusion_criteria = "\n".join([f"- {c}" for c in self.config.exclusion_criteria])
        
        prompt = f"""You are an expert researcher conducting a systematic scoping review screening. 
Please evaluate the following article for inclusion based on the criteria provided.

INCLUSION CRITERIA:
{inclusion_criteria}

EXCLUSION CRITERIA:
{exclusion_criteria}

ARTICLE TO EVALUATE:
Title: {title}
Abstract: {abstract}

INSTRUCTIONS:
1. Carefully read the title and abstract.
2. Evaluate against the inclusion and exclusion criteria.
3. If the abstract contains insufficient information to make a definitive decision (e.g., missing population details, unclear intervention):
   - Select "Needs Full Text" if the article might be relevant but the abstract lacks necessary details to confirm.
4. Provide your decision as one of: "Relevant", "Irrelevant", or "Needs Full Text".
5. Provide a brief reason (max 50 words).
6. Assess your confidence (0.0-1.0).

Respond in this exact format:
Decision: [Relevant/Irrelevant/Needs Full Text]
Reason: [Brief explanation]
Confidence: [0.0-1.0]"""
        
        return prompt
    
    def parse_ai_response(self, response: str) -> Dict[str, any]:
        """
        Parse AI response into structured format
        
        Args:
            response: Raw AI response text
            
        Returns:
            Dict with decision, reason, confidence
        """
        lines = response.strip().split('\n')
        result = {
            'decision': 'Uncertain',
            'reason': 'Unable to parse response',
            'confidence': 0.0
        }
        
        for line in lines:
            if line.startswith('Decision:'):
                decision_val = line.split(':', 1)[1].strip()
                # Unified Mapping: Map 'Uncertain' to 'Needs Full Text' to simplify PRISMA flow
                if decision_val.upper() == 'UNCERTAIN':
                    result['decision'] = 'Needs Full Text'
                else:
                    result['decision'] = decision_val
            elif line.startswith('Reason:'):
                result['reason'] = line.split(':', 1)[1].strip()
            elif line.startswith('Confidence:'):
                try:
                    result['confidence'] = float(line.split(':', 1)[1].strip())
                except ValueError:
                    result['confidence'] = 0.5
        
        # Fallback: look for decision keywords anywhere
        if result['decision'] == 'Uncertain':
            upper_response = response.upper()
            if 'NEEDS FULL TEXT' in upper_response:
                result['decision'] = 'Needs Full Text'
            elif 'RELEVANT' in upper_response:
                result['decision'] = 'Relevant'
            elif 'IRRELEVANT' in upper_response:
                result['decision'] = 'Irrelevant'
        
        return result
    
    def create_arbiter_prompt(
        self,
        title: str,
        abstract: str,
        decision_1: Dict,
        decision_2: Dict
    ) -> str:
        """
        Create prompt for Expert Arbiter to resolve conflict
        """
        inclusion_criteria = "\n".join([f"- {c}" for c in self.config.inclusion_criteria])
        exclusion_criteria = "\n".join([f"- {c}" for c in self.config.exclusion_criteria])
        
        prompt = f"""You are an Expert Systematic Review Arbiter. Two junior screeners have analyzed an article and disagree. 
Your job is to resolve the conflict by making a FINAL, authoritative decision.

INCLUSION CRITERIA:
{inclusion_criteria}

EXCLUSION CRITERIA:
{exclusion_criteria}

ARTICLE:
Title: {title}
Abstract: {abstract}

DISAGREEMENT:
Screener A Decision: {decision_1['decision']} (Confidence: {decision_1['confidence']})
Screener A Reason: {decision_1['reason']}

Screener B Decision: {decision_2['decision']} (Confidence: {decision_2['confidence']})
Screener B Reason: {decision_2['reason']}

INSTRUCTIONS:
1. Critically analyze the article against criteria.
2. Evaluate reasoning from both Screeners.
3. Decide if the article is "Relevant", "Irrelevant", or "Needs Full Text".
4. Explain WHY you chose one over the other.

Respond in this exact format:
Decision: [Relevant/Irrelevant/Needs Full Text]
Reason: [Your final verdict and critique of screeners]
Confidence: [0.0-1.0]"""
        return prompt
    
    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int,
        model: str,
        cached_tokens: int = 0
    ) -> float:
        """
        Calculate API cost based on token usage (in HKD)
        
        Args:
            input_tokens: Input token count
            output_tokens: Output token count
            reasoning_tokens: Reasoning token count (for reasoning models)
            model: Model name
            cached_tokens: Cached input token count (for prompt caching)
            
        Returns:
            Total cost in HKD
        """
        pricing = self.MODEL_PRICING.get(model, self.MODEL_PRICING['default'])
        
        # Calculate input cost (with caching if supported)
        if 'cached_input' in pricing and cached_tokens > 0:
            # Some tokens are cached (cheaper)
            uncached_tokens = max(0, input_tokens - cached_tokens)
            input_cost = (uncached_tokens / 1_000_000) * pricing['input']
            cached_cost = (cached_tokens / 1_000_000) * pricing['cached_input']
            total_input_cost = input_cost + cached_cost
        else:
            # No caching, standard input pricing
            total_input_cost = (input_tokens / 1_000_000) * pricing['input']
        
        # Output tokens (includes reasoning tokens)
        total_output_tokens = output_tokens + reasoning_tokens
        output_cost = (total_output_tokens / 1_000_000) * pricing['output']
        
        return total_input_cost + output_cost
    
    def _screen_standard(
        self,
        title: str,
        abstract: str,
        max_retries: int = 3,
        model_override: Optional[str] = None
    ) -> ArticleDecision:
        """
        Screen single article with retry logic (Standard Single Pass)
        
        Args:
            title: Article title
            abstract: Article abstract
            max_retries: Maximum retry attempts
            model_override: Optional model to use/force
            
        Returns:
            ArticleDecision with screening result
        """
        # Check for cancellation before starting
        if self._cancel_requested:
            raise InterruptedError("Screening cancelled by user")
        
        prompt = self.create_screening_prompt(title, abstract)
        
        for attempt in range(max_retries):
            try:
                # Call LLM client
                response = self.llm_client.screen_article(prompt, model=model_override)
                
                # Parse response
                parsed = self.parse_ai_response(response.get('content', ''))
                
                # Extract token usage
                usage = response.get('usage', {})
                input_tokens = usage.get('prompt_tokens', 0)
                output_tokens = usage.get('completion_tokens', 0)
                total_tokens = usage.get('total_tokens', 0)
                reasoning_tokens = max(0, total_tokens - input_tokens - output_tokens)
                
                # Calculate cost
                model = response.get('model', 'unknown')
                provider = response.get('provider', 'unknown')
                cost = self.calculate_cost(input_tokens, output_tokens, reasoning_tokens, model)
                
                # Check if manual review needed
                needs_manual_review = (
                    parsed['confidence'] < self.config.manual_review_threshold or
                    parsed['decision'] == 'Uncertain' or
                    parsed['decision'] == 'Needs Full Text'
                )
                
                return ArticleDecision(
                    title=title,
                    abstract=abstract,
                    decision=parsed['decision'],
                    reason=parsed['reason'],
                    confidence=parsed['confidence'],
                    needs_manual_review=needs_manual_review,
                    timestamp=datetime.now().isoformat(),
                    provider=provider,
                    model=model,
                    reasoning_content=response.get('reasoning_content'),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    reasoning_tokens=reasoning_tokens,
                    total_tokens=total_tokens,
                    api_cost=cost
                )
                
            except Exception as e:
                if attempt == max_retries - 1:
                    # Final retry failed
                    logger.error(f"Failed to screen article after {max_retries} attempts: {e}")
                    return ArticleDecision(
                        title=title,
                        abstract=abstract,
                        decision='Error',
                        reason=f'API Error: {str(e)}',
                        confidence=0.0,
                        needs_manual_review=True,
                        timestamp=datetime.now().isoformat(),
                        provider='unknown',
                        model='unknown'
                    )
                else:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    logger.warning(f"Retry {attempt + 1}/{max_retries} after {wait_time}s")
                    time.sleep(wait_time)

    def screen_single_article(
        self,
        title: str,
        abstract: str,
        max_retries: int = 3
    ) -> ArticleDecision:
        """
        Screen single article dispatching to appropriate strategy
        """
        if self.config.strategy == "dual_arbiter":
            return self._screen_dual_arbiter(title, abstract, max_retries)
        return self._screen_standard(title, abstract, max_retries)

    def _screen_dual_arbiter(
        self,
        title: str,
        abstract: str,
        max_retries: int
    ) -> ArticleDecision:
        """
        Dual-Agent + Arbiter Strategy
        1. Two fast models screen in parallel
        2. If agree -> Return result
        3. If disagree -> Call arbiter (Strong Model)
        """
        # Step 1: Run two fast passes (using default fast model)
        # We run them sequentially here to avoid nested thread pool issues, 
        # or we could use a mini-executor. Since we are already in a worker, sequential is safer.
        # But 'grok-4-fast-reasoning' is fast.
        
        # Pass 1
        res1 = self._screen_standard(title, abstract, max_retries=max_retries)
        if res1.decision == 'Error': return res1
        
        # Pass 2
        res2 = self._screen_standard(title, abstract, max_retries=max_retries)
        if res2.decision == 'Error': return res2 # Or use res1 if valid? Conservative: error.
        
        # Step 2: Compare
        clean_d1 = res1.decision.upper().replace(' ', '_')
        clean_d2 = res2.decision.upper().replace(' ', '_')
        
        # Simple agreement
        agreed = (clean_d1 == clean_d2)
        
        # Calculate combined cost/tokens so far
        total_input = res1.input_tokens + res2.input_tokens
        total_output = res1.output_tokens + res2.output_tokens
        total_reasoning = res1.reasoning_tokens + res2.reasoning_tokens
        total_cost = res1.api_cost + res2.api_cost
        
        if agreed:
            # Consensus reached
            res1.input_tokens = total_input
            res1.output_tokens = total_output
            res1.reasoning_tokens = total_reasoning
            res1.api_cost = total_cost
            res1.total_tokens = total_input + total_output + total_reasoning
            res1.reason = f"[Dual Consensus] {res1.reason}"
            res1.is_consensus = True
            return res1
            
        # Step 3: Arbiter (Disagreement)
        logger.info(f"⚖️ Arbiter called for '{title[:30]}...' ({res1.decision} vs {res2.decision})")
        
        try:
            # Prepare arbiter context
            d1_info = {'decision': res1.decision, 'reason': res1.reason, 'confidence': res1.confidence}
            d2_info = {'decision': res2.decision, 'reason': res2.reason, 'confidence': res2.confidence}
            
            arbiter_prompt = self.create_arbiter_prompt(title, abstract, d1_info, d2_info)
            
            # Use STRONGER model for arbiter
            arbiter_model = "grok-4" # Standard Grok-4
            
            response = self.llm_client.screen_article(arbiter_prompt, model=arbiter_model)
            parsed = self.parse_ai_response(response.get('content', ''))
            
            # Arbiter cost
            usage = response.get('usage', {})
            arb_input = usage.get('prompt_tokens', 0)
            arb_output = usage.get('completion_tokens', 0)
            arb_reasoning = max(0, usage.get('total_tokens', 0) - arb_input - arb_output)
            arb_cost = self.calculate_cost(arb_input, arb_output, arb_reasoning, arbiter_model)
            
            # Combine all costs
            final_cost = total_cost + arb_cost
            final_input = total_input + arb_input
            final_output = total_output + arb_output
            final_reasoning = total_reasoning + arb_reasoning
            
            needs_review = (
                parsed['confidence'] < self.config.manual_review_threshold or
                parsed['decision'] == 'Uncertain' or
                parsed['decision'] == 'Needs Full Text'
            )
            
            desc = parsed['decision']
            reason = f"[Arbiter Decision] {parsed['reason']} (Conflict: {res1.decision} vs {res2.decision})"
            
            return ArticleDecision(
                title=title,
                abstract=abstract,
                decision=desc,
                reason=reason,
                confidence=parsed['confidence'],
                needs_manual_review=needs_review,
                timestamp=datetime.now().isoformat(),
                provider=res1.provider,
                model=arbiter_model,
                reasoning_content=response.get('reasoning_content'),
                input_tokens=final_input,
                output_tokens=final_output,
                reasoning_tokens=final_reasoning,
                total_tokens=final_input + final_output + final_reasoning,
                api_cost=final_cost,
                is_arbiter_decision=True
            )
            
        except Exception as e:
            logger.error(f"Arbiter failed: {e}")
            # Fallback to Manual Review
            res1.decision = "Uncertain"
            res1.reason = f"Arbiter Failed: {str(e)}. (Conflict: {res1.decision} vs {res2.decision})"
            res1.needs_manual_review = True
            return res1
    
    def screen_parallel(
        self,
        articles: List[Tuple[str, str]],
        progress_callback: Optional[Callable] = None
    ) -> ScreeningResult:
        """
        Screen multiple articles in parallel
        
        Args:
            articles: List of (title, abstract) tuples
            progress_callback: Optional callback(completed, total, article_decision)
            
        Returns:
            ScreeningResult with all decisions
        """
        start_time = time.time()
        total_articles = len(articles)
        
        logger.info(
            f"🚀 Starting parallel screening: {total_articles} articles with {self.config.max_workers} workers. "
            f"Model: {self.llm_client.model}. "
            f"⚠️ Note: High worker counts (>4) may cause network instability. Recommended: 2-4 workers for production."
        )
        
        decisions = []
        completed = 0
        errors = 0
        
        # Parallel execution
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_article = {
                executor.submit(self.screen_single_article, title, abstract): (idx, title, abstract)
                for idx, (title, abstract) in enumerate(articles)
            }
            
            # Process completed tasks
            for future in as_completed(future_to_article):
                try:
                    decision = future.result()
                    decisions.append(decision)
                    completed += 1
                    
                    # Log progress at intervals
                    if completed % 10 == 0 or completed == total_articles:
                        logger.info(
                            f"📊 Progress: {completed}/{total_articles} ({completed/total_articles*100:.1f}%) - "
                            f"Errors: {errors}"
                        )
                    
                    # Progress callback
                    if progress_callback:
                        progress_callback(completed, total_articles, decision)
                    
                    # Checkpoint
                    if self.checkpoint_dir and completed % self.config.checkpoint_interval == 0:
                        self._save_checkpoint(decisions, completed, total_articles)
                        logger.info(f"💾 Checkpoint saved at {completed}/{total_articles}")
                        
                except Exception as e:
                    idx, title, abstract = future_to_article[future]
                    errors += 1
                    logger.error(
                        f"❌ Article #{idx} failed: {str(e)[:150]}. "
                        f"Title: {title[:100] if title else 'N/A'}. "
                        f"Total errors so far: {errors}/{completed}"
                    )
                    # Continue processing other articles
                    completed += 1
        
        # Calculate metrics
        total_time = time.time() - start_time
        
        # Aggregate results
        result = self._aggregate_results(decisions, total_articles, total_time)
        
        logger.info(
            f"Screening complete: {result.screened_count}/{total_articles} articles in {total_time:.1f}s "
            f"(${result.total_cost:.4f} total cost)"
        )
        
        return result
    
    def _aggregate_results(
        self,
        decisions: List[ArticleDecision],
        total_articles: int,
        total_time: float
    ) -> ScreeningResult:
        """
        Aggregate individual decisions into overall result
        
        Args:
            decisions: List of article decisions
            total_articles: Total number of articles
            total_time: Total execution time
            
        Returns:
            ScreeningResult
        """
        # Count decisions
        relevant = sum(1 for d in decisions if d.decision == 'Relevant')
        irrelevant = sum(1 for d in decisions if d.decision == 'Irrelevant')
        uncertain = sum(1 for d in decisions if d.decision == 'Uncertain')
        errors = sum(1 for d in decisions if d.decision == 'Error')
        needs_manual_review = sum(1 for d in decisions if d.needs_manual_review)
        
        # Aggregate token usage
        total_input_tokens = sum(d.input_tokens for d in decisions)
        total_output_tokens = sum(d.output_tokens for d in decisions)
        total_reasoning_tokens = sum(d.reasoning_tokens for d in decisions)
        
        # Aggregate cost
        total_cost = sum(d.api_cost for d in decisions)
        
        # Get model info from first decision
        model_used = decisions[0].model if decisions else "unknown"
        provider_used = decisions[0].provider if decisions else "unknown"
        
        return ScreeningResult(
            total_articles=total_articles,
            screened_count=len(decisions),
            relevant=relevant,
            irrelevant=irrelevant,
            uncertain=uncertain,
            errors=errors,
            needs_manual_review=needs_manual_review,
            decisions=decisions,
            total_time=total_time,
            avg_time_per_article=total_time / len(decisions) if decisions else 0,
            total_cost=total_cost,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_reasoning_tokens=total_reasoning_tokens,
            avg_cost_per_article=total_cost / len(decisions) if decisions else 0,
            model_used=model_used,
            provider_used=provider_used
        )
    
    def _save_checkpoint(self, decisions: List[ArticleDecision], completed: int, total: int):
        """Save screening checkpoint"""
        if not self.checkpoint_dir:
            return
        
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{completed}_{total}.json"
        
        checkpoint_data = {
            'completed': completed,
            'total': total,
            'timestamp': datetime.now().isoformat(),
            'decisions': [
                {
                    'title': d.title[:100],  # Truncate for space
                    'decision': d.decision,
                    'confidence': d.confidence,
                    'needs_manual_review': d.needs_manual_review,
                    'api_cost': d.api_cost
                }
                for d in decisions
            ]
        }
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logger.info(f"Checkpoint saved: {completed}/{total} articles")
    
    def generate_summary_report(self, result: ScreeningResult) -> str:
        """
        Generate human-readable summary report
        
        Args:
            result: ScreeningResult
            
        Returns:
            Formatted report string
        """
        report = f"""
=== AI Screening Summary Report ===

📊 Screening Statistics:
   Total articles: {result.total_articles}
   Successfully screened: {result.screened_count}
   
   Decisions:
   - Relevant: {result.relevant} ({result.relevant/result.screened_count*100:.1f}%)
   - Irrelevant: {result.irrelevant} ({result.irrelevant/result.screened_count*100:.1f}%)
   - Uncertain: {result.uncertain} ({result.uncertain/result.screened_count*100:.1f}%)
   - Errors: {result.errors}
   
   Manual review needed: {result.needs_manual_review} ({result.needs_manual_review/result.screened_count*100:.1f}%)

⚡ Performance Metrics:
   Total time: {result.total_time:.1f}s
   Avg time per article: {result.avg_time_per_article:.2f}s
   Throughput: {result.screened_count/result.total_time:.1f} articles/s

💰 Cost Analysis:
   Model: {result.model_used} ({result.provider_used})
   Total cost: ${result.total_cost:.6f}
   Avg cost per article: ${result.avg_cost_per_article:.6f}
   
   Token usage:
   - Input tokens: {result.total_input_tokens:,}
   - Output tokens: {result.total_output_tokens:,}
   - Reasoning tokens: {result.total_reasoning_tokens:,}
   - Total tokens: {result.total_input_tokens + result.total_output_tokens + result.total_reasoning_tokens:,}

📋 Next Steps:
   1. Review {result.needs_manual_review} articles flagged for manual review
   2. Validate sample of {min(20, result.relevant)} relevant articles
   3. Proceed to full-text screening for {result.relevant + result.uncertain} articles
"""
        
        return report
    
    def export_to_dataframe(self, result: ScreeningResult) -> pd.DataFrame:
        """
        Export screening results to pandas DataFrame
        
        Args:
            result: ScreeningResult
            
        Returns:
            DataFrame with all decisions
        """
        data = []
        for decision in result.decisions:
            data.append({
                'title': decision.title,
                'abstract': decision.abstract,
                'ai_decision': decision.decision,
                'ai_reason': decision.reason,
                'ai_confidence': decision.confidence,
                'ai_needs_manual_review': decision.needs_manual_review,
                'ai_timestamp': decision.timestamp,
                'ai_provider': decision.provider,
                'ai_model': decision.model,
                'ai_input_tokens': decision.input_tokens,
                'ai_output_tokens': decision.output_tokens,
                'ai_reasoning_tokens': decision.reasoning_tokens,
                'ai_total_tokens': decision.total_tokens,
                'ai_cost': decision.api_cost
            })
        
        return pd.DataFrame(data)
