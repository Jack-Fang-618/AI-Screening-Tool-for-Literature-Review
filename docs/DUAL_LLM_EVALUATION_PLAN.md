# 🤖 Dual-LLM Deliberative Evaluation System - Implementation Plan

**Version:** 1.0
**Date:** November 11, 2025
**Status:** Planning Phase
**Purpose:** Implement LLM-to-LLM discussion mechanism for consensus-based screening quality assurance

---

## 📋 Executive Summary

### Vision

Create an advanced evaluation system where two LLMs engage in structured dialogue to validate and refine screening decisions. When they disagree, they enter a discussion loop (max 3 rounds) to reach consensus or flag for human review.

### Key Innovation

**Traditional Approach:**

```
Single LLM → Decision → Done
```

**Our Approach:**

```
Primary LLM → Decision
              ↓
Evaluator LLM → Review
              ↓
     ┌────────┴────────┐
     ▼                 ▼
  Agree            Disagree
     │                 │
  Accept         Discussion Loop
     │              (2-3 rounds)
     │                 │
     └────────┬────────┘
              ▼
    Final Decision or Human Review
```

### Three Possible Outcomes

1. **Agree** → High confidence, accept result immediately
2. **Disagree** → Enter discussion loop, LLMs debate and refine (max 3 rounds)
3. **Human Needed** → Abstract unclear or criteria ambiguous, flag for manual review

---

## 📐 Section 1: System Architecture & Data Models

### 1.1 Conceptual Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ARTICLE + CRITERIA INPUT                          │
│              (Title, Abstract, Inclusion/Exclusion Criteria)         │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PRIMARY LLM (Screener)                            │
│                    Model: grok-4-fast-reasoning                      │
│                                                                       │
│  Task: Initial screening decision                                    │
│  Output: {decision, reasoning, confidence}                           │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 EVALUATION LLM (Quality Checker)                     │
│                    Model: grok-4-fast-reasoning                      │
│                                                                       │
│  Input: Article + Criteria + Primary Decision                        │
│  Task: Evaluate decision quality using rubric                        │
│  Output: {                                                           │
│    evaluation_result: "agree" | "disagree" | "human_needed",        │
│    reasoning: "Why agree/disagree",                                  │
│    concerns: ["List of specific concerns"],                          │
│    suggestion: "What primary LLM should reconsider"                  │
│  }                                                                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
         ┌─────────┐   ┌──────────┐   ┌────────────────────┐
         │ AGREE   │   │ DISAGREE │   │   HUMAN NEEDED     │
         │         │   │          │   │ (Abstract unclear/ │
         └────┬────┘   └─────┬────┘   │ Criteria ambiguous)│
              │              │         └─────────┬──────────┘
              │              │                      │
              ▼              ▼                      ▼
         ┌─────────┐   ┌──────────────────────┐  ┌──────────────┐
         │ ACCEPT  │   │  DISCUSSION LOOP     │  │ FLAG FOR     │
         │ RESULT  │   │  (Max 3 iterations)  │  │ HUMAN REVIEW │
         └─────────┘   │                      │  │ (Skip loop)  │
                       │  ┌───────────────────┐  └──────────────┘
                       │  │ Round 1:           │  │
                       │  │ - Eval gives feedback │  │
                       │  │ - Primary reconsiders │  │
                       │  │ - Check agreement     │  │
                       │  └───────────┬───────────┘  │
                       │              │ No consensus  │
                       │  ┌───────────▼────────────┐  │
                       │  │ Round 2:               │  │
                       │  │ - More detailed debate │  │
                       │  │ - Primary adjusts      │  │
                       │  │ - Re-evaluate          │  │
                       │  └───────────┬────────────┘  │
                       │              │ No consensus  │
                       │  ┌───────────▼────────────┐  │
                       │  │ Round 3 (Final):       │  │
                       │  │ - Final arguments      │  │
                       │  │ - Force decision or    │  │
                       │  │ - Flag for human       │  │
                       │  └────────────────────────┘  │
                       └──────────────┬───────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────┐
                       │   FINAL OUTPUT:              │
                       │   - Consensus decision       │
                       │   - Discussion history       │
                       │   - Confidence score         │
                       │   - Human review flag        │
                       └──────────────────────────────┘
```

### 1.2 Database Schema Design

**New Tables to Add:**

```sql
-- Table 1: Evaluation Sessions
CREATE TABLE evaluation_sessions (
    id TEXT PRIMARY KEY,                    -- UUID
    screening_result_id TEXT NOT NULL,      -- FK to screening_results
    article_title TEXT NOT NULL,
    article_abstract TEXT NOT NULL,
    criteria TEXT NOT NULL,                 -- JSON: {inclusion: [], exclusion: []}
  
    -- Primary LLM output
    primary_decision TEXT NOT NULL,         -- "include" | "exclude"
    primary_reasoning TEXT NOT NULL,
    primary_confidence REAL NOT NULL,
  
    -- Evaluation result
    evaluation_result TEXT NOT NULL,        -- "agree" | "disagree" | "human_needed"
  
    -- Final outcome
    final_decision TEXT NOT NULL,           -- Final consensus decision
    final_confidence REAL NOT NULL,         -- Final confidence (0-1)
    requires_human_review BOOLEAN NOT NULL, -- Flag for manual review
    discussion_rounds INTEGER DEFAULT 0,    -- Number of discussion rounds (0-3)
  
    -- Metadata
    total_cost REAL DEFAULT 0.0,           -- Total API cost (HKD)
    processing_time REAL DEFAULT 0.0,      -- Total time (seconds)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  
    FOREIGN KEY (screening_result_id) REFERENCES screening_results(id)
);

-- Table 2: Discussion History (for iterative loop)
CREATE TABLE discussion_rounds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_session_id TEXT NOT NULL,
    round_number INTEGER NOT NULL,         -- 1, 2, or 3
  
    -- Round input
    input_decision TEXT NOT NULL,          -- Current decision being evaluated
    input_reasoning TEXT NOT NULL,         -- Current reasoning
  
    -- Evaluator response
    evaluator_result TEXT NOT NULL,        -- "agree" | "disagree" | "human_needed"
    evaluator_reasoning TEXT NOT NULL,     -- Why evaluator agrees/disagrees
    evaluator_concerns TEXT,               -- JSON: ["concern1", "concern2", ...]
    evaluator_suggestion TEXT,             -- Specific suggestion for reconsideration
  
    -- Primary LLM response (if disagreed)
    primary_response TEXT,                 -- New reasoning after reconsideration
    primary_new_decision TEXT,             -- New decision (may change)
    primary_new_confidence REAL,           -- New confidence
  
    -- Round outcome
    consensus_reached BOOLEAN NOT NULL,    -- Did they agree this round?
  
    -- Metadata
    round_cost REAL DEFAULT 0.0,
    round_time REAL DEFAULT 0.0,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  
    FOREIGN KEY (evaluation_session_id) REFERENCES evaluation_sessions(id)
);

-- Table 3: Evaluation Rubric Scores
CREATE TABLE rubric_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_session_id TEXT NOT NULL,
  
    -- Rubric criteria (0-5 scale each)
    relevance_score REAL NOT NULL,         -- How relevant to criteria
    reasoning_quality_score REAL NOT NULL, -- Quality of reasoning
    evidence_score REAL NOT NULL,          -- Supporting evidence from abstract
    consistency_score REAL NOT NULL,       -- Internal consistency
    completeness_score REAL NOT NULL,      -- Completeness of analysis
  
    -- Weighted scores
    weighted_total REAL NOT NULL,          -- Sum of weighted scores
  
    -- Evaluator notes
    relevance_notes TEXT,
    reasoning_notes TEXT,
    evidence_notes TEXT,
    consistency_notes TEXT,
    completeness_notes TEXT,
  
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  
    FOREIGN KEY (evaluation_session_id) REFERENCES evaluation_sessions(id)
);
```

### 1.3 Pydantic Data Models

**File: `backend/models/evaluation.py`** (NEW)

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime
from enum import Enum

class EvaluationResult(str, Enum):
    """Possible evaluation outcomes"""
    AGREE = "agree"
    DISAGREE = "disagree"
    HUMAN_NEEDED = "human_needed"

class DecisionType(str, Enum):
    """Screening decisions"""
    INCLUDE = "include"
    EXCLUDE = "exclude"
    UNCERTAIN = "uncertain"

# ============================================
# Core Models
# ============================================

class PrimaryDecision(BaseModel):
    """Primary LLM screening decision"""
    decision: DecisionType
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)
    processing_time: float = 0.0
    cost: float = 0.0

class RubricScore(BaseModel):
    """Evaluation rubric scores"""
    relevance_score: float = Field(ge=0.0, le=5.0, description="Relevance to criteria")
    reasoning_quality_score: float = Field(ge=0.0, le=5.0, description="Quality of reasoning")
    evidence_score: float = Field(ge=0.0, le=5.0, description="Supporting evidence")
    consistency_score: float = Field(ge=0.0, le=5.0, description="Internal consistency")
    completeness_score: float = Field(ge=0.0, le=5.0, description="Completeness of analysis")
  
    # Notes for each criterion
    relevance_notes: Optional[str] = None
    reasoning_notes: Optional[str] = None
    evidence_notes: Optional[str] = None
    consistency_notes: Optional[str] = None
    completeness_notes: Optional[str] = None
  
    # Weights (should sum to 1.0)
    WEIGHTS = {
        'relevance': 0.35,
        'reasoning_quality': 0.25,
        'evidence': 0.20,
        'consistency': 0.15,
        'completeness': 0.05
    }
  
    @property
    def weighted_total(self) -> float:
        """Calculate weighted total score"""
        return (
            self.relevance_score * self.WEIGHTS['relevance'] +
            self.reasoning_quality_score * self.WEIGHTS['reasoning_quality'] +
            self.evidence_score * self.WEIGHTS['evidence'] +
            self.consistency_score * self.WEIGHTS['consistency'] +
            self.completeness_score * self.WEIGHTS['completeness']
        )

class EvaluatorResponse(BaseModel):
    """Evaluator LLM response"""
    result: EvaluationResult
    reasoning: str
    concerns: List[str] = Field(default_factory=list)
    suggestion: Optional[str] = None
    rubric_scores: RubricScore
    confidence: float = Field(ge=0.0, le=1.0)
    processing_time: float = 0.0
    cost: float = 0.0

class DiscussionRound(BaseModel):
    """Single round of discussion between LLMs"""
    round_number: int = Field(ge=1, le=3)
  
    # Input to this round
    input_decision: DecisionType
    input_reasoning: str
    input_confidence: float
  
    # Evaluator response
    evaluator_response: EvaluatorResponse
  
    # Primary LLM counter-response (if disagreed)
    primary_response: Optional[str] = None
    primary_new_decision: Optional[DecisionType] = None
    primary_new_confidence: Optional[float] = None
  
    # Round outcome
    consensus_reached: bool = False
    round_cost: float = 0.0
    round_time: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)

class EvaluationSession(BaseModel):
    """Complete evaluation session with discussion history"""
    id: str
    screening_result_id: str
  
    # Article info
    article_title: str
    article_abstract: str
    criteria: dict  # {inclusion: [], exclusion: []}
  
    # Primary decision
    primary_decision: PrimaryDecision
  
    # Discussion rounds (0-3)
    discussion_rounds: List[DiscussionRound] = Field(default_factory=list)
  
    # Final outcome
    final_decision: DecisionType
    final_confidence: float = Field(ge=0.0, le=1.0)
    final_reasoning: str
    requires_human_review: bool = False
  
    # Metadata
    total_cost: float = 0.0
    processing_time: float = 0.0
    created_at: datetime = Field(default_factory=datetime.now)

# ============================================
# API Request/Response Models
# ============================================

class StartEvaluationRequest(BaseModel):
    """Request to start evaluation"""
    screening_result_id: str
    max_discussion_rounds: int = Field(default=3, ge=1, le=5)
    enable_discussion: bool = True  # If False, single evaluation only
  
class EvaluationProgressResponse(BaseModel):
    """Progress update during evaluation"""
    evaluation_session_id: str
    status: Literal["running", "completed", "failed"]
    current_round: int
    max_rounds: int
    consensus_reached: bool
    estimated_time_remaining: Optional[float] = None
    current_cost: float = 0.0

class EvaluationResultResponse(BaseModel):
    """Final evaluation result"""
    session: EvaluationSession
    summary: dict  # Summary statistics
    visualization_data: Optional[dict] = None  # Data for charts
```

### 1.4 Configuration Settings

**File: `backend/core/evaluation_config.py`** (NEW)

```python
from dataclasses import dataclass
from typing import Dict

@dataclass
class EvaluationConfig:
    """Configuration for evaluation system"""
  
    # Models
    primary_model: str = "grok-4-fast-reasoning"
    evaluator_model: str = "grok-4-fast-reasoning"
  
    # Discussion loop settings
    max_discussion_rounds: int = 3
    enable_discussion: bool = True
    consensus_threshold: float = 0.8  # Agreement score needed for consensus
  
    # Rubric weights (must sum to 1.0)
    rubric_weights: Dict[str, float] = None
  
    # Confidence thresholds
    high_confidence_threshold: float = 0.8  # Auto-accept
    low_confidence_threshold: float = 0.6   # Auto-review
  
    # Cost control
    max_cost_per_article: float = 0.50  # HKD
  
    # Timeouts
    llm_timeout_seconds: int = 30
    max_total_time_seconds: int = 120
  
    def __post_init__(self):
        if self.rubric_weights is None:
            self.rubric_weights = {
                'relevance': 0.35,
                'reasoning_quality': 0.25,
                'evidence': 0.20,
                'consistency': 0.15,
                'completeness': 0.05
            }
    
        # Validate weights sum to 1.0
        total = sum(self.rubric_weights.values())
        if not (0.99 <= total <= 1.01):  # Allow small floating point error
            raise ValueError(f"Rubric weights must sum to 1.0, got {total}")

# Default configuration
DEFAULT_EVAL_CONFIG = EvaluationConfig()
```

### 1.5 Database Migration Script

**File: `backend/db/migrations/add_evaluation_tables.py`** (NEW)

```python
"""
Migration: Add evaluation system tables
Run with: python -m backend.db.migrations.add_evaluation_tables
"""

from backend.db.config import get_engine
from sqlalchemy import text

def upgrade():
    """Add evaluation tables"""
    engine = get_engine()
  
    with engine.begin() as conn:
        # Evaluation sessions table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS evaluation_sessions (
                id TEXT PRIMARY KEY,
                screening_result_id TEXT NOT NULL,
                article_title TEXT NOT NULL,
                article_abstract TEXT NOT NULL,
                criteria TEXT NOT NULL,
            
                primary_decision TEXT NOT NULL,
                primary_reasoning TEXT NOT NULL,
                primary_confidence REAL NOT NULL,
            
                evaluation_result TEXT NOT NULL,
            
                final_decision TEXT NOT NULL,
                final_confidence REAL NOT NULL,
                requires_human_review BOOLEAN NOT NULL,
                discussion_rounds INTEGER DEFAULT 0,
            
                total_cost REAL DEFAULT 0.0,
                processing_time REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
                FOREIGN KEY (screening_result_id) REFERENCES screening_results(id)
            )
        """))
    
        # Discussion rounds table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS discussion_rounds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_session_id TEXT NOT NULL,
                round_number INTEGER NOT NULL,
            
                input_decision TEXT NOT NULL,
                input_reasoning TEXT NOT NULL,
            
                evaluator_result TEXT NOT NULL,
                evaluator_reasoning TEXT NOT NULL,
                evaluator_concerns TEXT,
                evaluator_suggestion TEXT,
            
                primary_response TEXT,
                primary_new_decision TEXT,
                primary_new_confidence REAL,
            
                consensus_reached BOOLEAN NOT NULL,
            
                round_cost REAL DEFAULT 0.0,
                round_time REAL DEFAULT 0.0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
                FOREIGN KEY (evaluation_session_id) REFERENCES evaluation_sessions(id)
            )
        """))
    
        # Rubric scores table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS rubric_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_session_id TEXT NOT NULL,
            
                relevance_score REAL NOT NULL,
                reasoning_quality_score REAL NOT NULL,
                evidence_score REAL NOT NULL,
                consistency_score REAL NOT NULL,
                completeness_score REAL NOT NULL,
            
                weighted_total REAL NOT NULL,
            
                relevance_notes TEXT,
                reasoning_notes TEXT,
                evidence_notes TEXT,
                consistency_notes TEXT,
                completeness_notes TEXT,
            
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
                FOREIGN KEY (evaluation_session_id) REFERENCES evaluation_sessions(id)
            )
        """))
    
        print("✅ Evaluation tables created successfully")

def downgrade():
    """Remove evaluation tables"""
    engine = get_engine()
  
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS rubric_scores"))
        conn.execute(text("DROP TABLE IF EXISTS discussion_rounds"))
        conn.execute(text("DROP TABLE IF EXISTS evaluation_sessions"))
    
        print("✅ Evaluation tables removed successfully")

if __name__ == "__main__":
    import sys
  
    if len(sys.argv) > 1 and sys.argv[1] == "downgrade":
        downgrade()
    else:
        upgrade()
```

### 1.6 Implementation Checklist

**Phase 1: Database Schema**

- [ ] Create migration script
- [ ] Test table creation
- [ ] Verify foreign key constraints
- [ ] Add indexes for performance

**Phase 2: Pydantic Models**

- [ ] Define all data models
- [ ] Add validation rules
- [ ] Write model tests
- [ ] Document model fields

**Phase 3: Configuration**

- [ ] Create evaluation config
- [ ] Define default settings
- [ ] Add config validation

**Phase 4: Documentation**

- [ ] Document architecture
- [ ] Create data flow diagrams
- [ ] Write usage examples

**Phase 5: Testing**

- [ ] Write unit tests for models
- [ ] Test database operations
- [ ] Validate configuration
- [ ] Integration tests

---

## 🔄 Implementation Sections Overview

1. ✅ **Section 1: System Architecture & Data Models** - COMPLETE
2. ✅ **Section 2: Core Evaluation Engine** - COMPLETE
3. ✅ **Section 3: Discussion Loop Mechanism** - COMPLETE
4. ✅ **Section 4: Frontend "Evaluation" Page** - COMPLETE
5. ✅ **Section 5: Testing & Deployment Strategy** - COMPLETE
6. ✅ **Section 6: Implementation Priorities & Dependencies** - COMPLETE
7. ✅ **Section 7: Cost-Benefit Analysis** - COMPLETE

---

## 🧠 Section 2: Core Evaluation Engine

### 2.1 Evaluator LLM Prompt Design

**Purpose:** The Evaluator LLM reviews the Primary LLM's screening decision and provides structured quality assessment.

**Prompt Structure:**

```
ROLE: You are an expert quality control reviewer for systematic literature reviews.

TASK: Evaluate the screening decision made by another AI reviewer.

INPUT PROVIDED:
1. Article Title: [title]
2. Article Abstract: [abstract]
3. Screening Criteria:
   - Inclusion: [list]
   - Exclusion: [list]
4. Primary Reviewer's Decision: [Include/Exclude]
5. Primary Reviewer's Reasoning: [reasoning text]
6. Primary Reviewer's Confidence: [0.0-1.0]

YOUR EVALUATION TASK:
Assess the quality and correctness of the primary reviewer's decision using these criteria:

1. RELEVANCE (35% weight): Does the article match the screening criteria?
   - Score 0-5: How well does it align with inclusion/exclusion criteria?
   - Note specific criteria that are met or violated

2. REASONING QUALITY (25% weight): Is the reasoning logical and well-supported?
   - Score 0-5: Quality of argumentation
   - Identify logical gaps or strong points

3. EVIDENCE (20% weight): Does the reasoning cite specific evidence from the abstract?
   - Score 0-5: Use of concrete evidence vs vague statements
   - Note what evidence was used or missing

4. CONSISTENCY (15% weight): Is the decision internally consistent with the reasoning?
   - Score 0-5: Does the decision match the reasoning provided?
   - Flag any contradictions

5. COMPLETENESS (5% weight): Did the review address all relevant criteria?
   - Score 0-5: Coverage of all inclusion/exclusion points
   - Note what was overlooked

OUTPUT FORMAT (JSON):
{
  "evaluation_result": "agree" | "disagree" | "human_needed",
  "reasoning": "Your overall assessment in 2-3 sentences",
  "rubric_scores": {
    "relevance_score": [0-5],
    "relevance_notes": "specific comments",
    "reasoning_quality_score": [0-5],
    "reasoning_notes": "specific comments",
    "evidence_score": [0-5],
    "evidence_notes": "specific comments",
    "consistency_score": [0-5],
    "consistency_notes": "specific comments",
    "completeness_score": [0-5],
    "completeness_notes": "specific comments"
  },
  "concerns": ["concern 1", "concern 2", ...],
  "suggestion": "What the primary reviewer should reconsider (if disagree)",
  "confidence": [0.0-1.0]
}

DECISION LOGIC:
- "agree": Scores ≥4.0 average, decision is sound
- "disagree": Scores ≥3.0 but decision seems incorrect, can be improved through discussion
- "human_needed": Scores <3.0 OR abstract is too vague OR criteria are ambiguous
```

**Key Design Principles:**
- Structured rubric prevents arbitrary disagreement
- Forces evaluator to cite specific evidence
- Three-tier decision prevents unnecessary loops
- JSON output ensures parseable results

### 2.2 Rubric Scoring Logic

**Weighted Scoring Formula:**

```
Weighted Total = 
  (Relevance × 0.35) +
  (Reasoning Quality × 0.25) +
  (Evidence × 0.20) +
  (Consistency × 0.15) +
  (Completeness × 0.05)

Final Score Range: 0.0 - 5.0
```

**Decision Thresholds:**

| Weighted Average | Evaluation Result | Action |
|-----------------|-------------------|---------|
| 4.0 - 5.0 | AGREE | Accept primary decision |
| 3.0 - 3.9 | DISAGREE | Enter discussion loop |
| 0.0 - 2.9 | HUMAN_NEEDED | Flag for manual review |

**Special Cases:**

1. **Abstract Quality Issues:**
   - If abstract is <50 words → HUMAN_NEEDED
   - If abstract is mostly boilerplate → HUMAN_NEEDED
   - Missing key information (methods, results) → HUMAN_NEEDED

2. **Criteria Ambiguity:**
   - Inclusion/exclusion criteria conflict → HUMAN_NEEDED
   - Criteria too vague to apply → HUMAN_NEEDED

### 2.3 Single-Pass Evaluation Flow

**Architecture Overview:**

```
┌──────────────────────────────────────┐
│  Input: Primary Decision             │
│  - decision, reasoning, confidence   │
└─────────────┬────────────────────────┘
              │
              ▼
┌──────────────────────────────────────┐
│  Evaluator LLM API Call              │
│  - Prompt with rubric                │
│  - Parse JSON response               │
│  - Track tokens & cost               │
└─────────────┬────────────────────────┘
              │
              ▼
┌──────────────────────────────────────┐
│  Calculate Weighted Score            │
│  - Apply rubric weights              │
│  - Determine threshold tier          │
└─────────────┬────────────────────────┘
              │
              ▼
┌──────────────────────────────────────┐
│  Evaluation Result                   │
│  - AGREE / DISAGREE / HUMAN_NEEDED   │
│  - Store in database                 │
└──────────────────────────────────────┘
```

**Component Responsibilities:**

**ScreeningEvaluator Class:**
- Manages evaluator LLM client
- Constructs evaluation prompts
- Parses and validates responses
- Calculates weighted scores
- Determines evaluation result

**Key Methods:**
- `evaluate_decision()` → Main evaluation entry point
- `_build_prompt()` → Construct evaluator prompt
- `_parse_response()` → Parse JSON response
- `_calculate_weighted_score()` → Apply rubric weights
- `_determine_result()` → Map score to result type

### 2.4 API Cost Tracking

**Token Usage Breakdown:**

```
Evaluator Call Tokens:
├─ Input Tokens:
│  ├─ System prompt: ~300 tokens
│  ├─ Article (title + abstract): ~200-500 tokens
│  ├─ Criteria: ~100-200 tokens
│  ├─ Primary decision + reasoning: ~150-300 tokens
│  └─ Total Input: ~750-1,300 tokens
│
├─ Output Tokens:
│  ├─ Evaluation reasoning: ~100-200 tokens
│  ├─ Rubric scores + notes: ~150-250 tokens
│  ├─ JSON structure: ~50 tokens
│  └─ Total Output: ~300-500 tokens
│
└─ Total per Evaluation: ~1,050-1,800 tokens
```

**Cost Calculation (HKD):**

```
Model: grok-4-fast-reasoning
Input: HKD $1.556 per 1M tokens
Output: HKD $3.89 per 1M tokens

Per Evaluation Cost:
= (1,300 × $1.556 / 1,000,000) + (500 × $3.89 / 1,000,000)
= $0.002 + $0.002
= ~HKD $0.004 per evaluation

For 5,000 articles:
Single-LLM screening: 5,000 × $0.05 = $250
Dual-LLM (all evaluated): 5,000 × ($0.05 + $0.004) = $270
Overhead: ~8% cost increase
```

**Cost Optimization Strategies:**

1. **Selective Evaluation:**
   - Only evaluate borderline decisions (confidence 0.6-0.8)
   - Reduces evaluations by ~60%

2. **Prompt Caching:**
   - Cache system prompt and criteria
   - Reduces input tokens by ~30%

3. **Batch Processing:**
   - Process multiple evaluations in parallel
   - Reduces wall-clock time, not cost

---

## 🔄 Section 3: Discussion Loop Mechanism

### 3.1 Iterative Discussion Protocol

**Loop Workflow:**

```
Round 1: Initial Response
├─ Evaluator provides concerns & suggestion
├─ Primary LLM receives feedback
├─ Primary LLM reconsiders decision
└─ Check for consensus → If YES: Exit | If NO: Continue

Round 2: Deeper Analysis
├─ Evaluator elaborates on concerns
├─ Primary LLM provides counter-argument or adjustment
├─ Both assess specific evidence points
└─ Check for consensus → If YES: Exit | If NO: Continue

Round 3: Final Deliberation
├─ Evaluator gives final assessment
├─ Primary LLM makes final decision
├─ Force conclusion: Accept last decision or flag for human
└─ Exit (max rounds reached)
```

**Round Structure:**

Each discussion round consists of:

1. **Input State:**
   - Current decision
   - Current reasoning
   - Previous round feedback (if applicable)

2. **Evaluator Turn:**
   - Reviews current position
   - Identifies specific issues
   - Suggests what to reconsider
   - Provides concrete examples

3. **Primary LLM Turn:**
   - Responds to concerns
   - Either: Adjusts decision with new reasoning
   - Or: Defends current decision with more evidence

4. **Consensus Check:**
   - Calculate agreement score
   - If ≥80% aligned → Consensus reached
   - If <80% → Continue to next round

### 3.2 Consensus Detection Algorithm

**Agreement Calculation:**

```
Agreement Score = 
  0.40 × Decision Match (1.0 if same, 0.0 if different) +
  0.30 × Confidence Alignment (1.0 - |conf1 - conf2|) +
  0.30 × Rubric Score Alignment (weighted_score / 5.0)

Threshold: ≥0.80 = Consensus Reached
```

**Decision Match Criteria:**

| Primary Decision | Evaluator Assessment | Match Score |
|-----------------|---------------------|-------------|
| Include | AGREE | 1.0 |
| Exclude | AGREE | 1.0 |
| Include | DISAGREE (says should exclude) | 0.0 |
| Exclude | DISAGREE (says should include) | 0.0 |

**Confidence Alignment:**

```
Primary Confidence: 0.85
Evaluator Confidence: 0.90
Difference: |0.85 - 0.90| = 0.05
Alignment: 1.0 - 0.05 = 0.95 (very aligned)

Primary Confidence: 0.70
Evaluator Confidence: 0.40
Difference: |0.70 - 0.40| = 0.30
Alignment: 1.0 - 0.30 = 0.70 (moderately aligned)
```

### 3.3 Loop Termination Conditions

**Termination Triggers:**

1. **Consensus Reached:**
   - Agreement score ≥0.80
   - Exit immediately with consensus decision
   - Store all round history

2. **Max Rounds Reached (3 rounds):**
   - No consensus after 3 rounds
   - Final decision = Last primary decision
   - Flag for human review
   - Store complete discussion history

3. **Human Review Needed:**
   - Evaluator determines HUMAN_NEEDED at any round
   - Exit loop immediately
   - Flag for manual review
   - No further rounds

4. **Cost Limit Exceeded:**
   - Total cost exceeds configured max
   - Exit with current decision
   - Flag for review

**State Management:**

Each round stores:
- Round number (1-3)
- Input decision & reasoning
- Evaluator feedback
- Primary response
- Agreement score
- Consensus status
- Tokens & cost

**Persistence:**

All rounds saved to `discussion_rounds` table:
- Enables review of full dialogue
- Supports ML training on successful consensus patterns
- Audit trail for quality control

### 3.4 Prompt Evolution Across Rounds

**Round 1 Prompt (Evaluator to Primary):**

```
The evaluator has concerns about your decision:

Concerns:
- [concern 1]
- [concern 2]

Suggestion: [what to reconsider]

Please reconsider your decision. You may:
1. Maintain your decision with additional supporting evidence
2. Adjust your decision based on the feedback

Provide:
- Your decision (include/exclude)
- Updated reasoning (addressing the concerns)
- Confidence (0-1)
```

**Round 2 Prompt (After Primary Response):**

```
The primary reviewer responded:
"[primary response]"

New decision: [include/exclude]
New reasoning: [reasoning]

Evaluate this updated position:
- Have your concerns been adequately addressed?
- Is there stronger evidence now?
- Do you agree or still disagree?
```

**Round 3 Prompt (Final Round):**

```
This is the final round of discussion.

Primary reviewer's final position:
[decision + reasoning]

Make your final assessment:
- AGREE if concerns are resolved
- DISAGREE if fundamental issues remain
- HUMAN_NEEDED if this requires expert judgment

This decision is final and will be recorded.
```

---

## 🎨 Section 4: Frontend "Evaluation" Page

### 4.1 Page Layout & Navigation

**Page Structure:**

```
┌────────────────────────────────────────────────────────────┐
│  🔍 Evaluation                                   [Settings] │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  Select Screening Session                            │  │
│  │  [Dropdown: Task ID | Date | # Articles]            │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  Evaluation Mode                                     │  │
│  │  ○ Evaluate All Results                             │  │
│  │  ○ Evaluate Only Borderline (0.6-0.8 confidence)    │  │
│  │  ○ Evaluate Selected Articles                       │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
│  [Start Evaluation]                                        │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

**Navigation:**
- Accessible from Results page
- Link in main sidebar
- "Evaluate Results" button on completed screening tasks

### 4.2 Real-Time Discussion Viewer

**Live Discussion Display:**

```
┌────────────────────────────────────────────────────────────┐
│  Article: "Effects of Continuous Glucose Monitoring..."    │
│  Status: ⚡ Discussion Round 2/3                           │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ 🤖 Primary LLM                          Confidence: 0.75│
│  │                                                       │  │
│  │ Decision: INCLUDE                                    │  │
│  │ Reasoning: The study investigates CGM in pediatric  │  │
│  │ T1D patients, matching our population criteria...   │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ 🔍 Evaluator LLM                        Round 1      │  │
│  │                                                       │  │
│  │ Result: DISAGREE                                     │  │
│  │ Concerns:                                            │  │
│  │  • Age range not clearly specified in abstract      │  │
│  │  • No mention of glycemic control outcomes          │  │
│  │ Suggestion: Verify age criteria match 5-12 years    │  │
│  │                                                       │  │
│  │ Rubric Scores:                                       │  │
│  │  Relevance: 3.5/5 ⚠️                                │  │
│  │  Reasoning: 4.0/5 ✓                                 │  │
│  │  Evidence: 3.0/5 ⚠️                                 │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ 🤖 Primary LLM Response                 Round 2      │  │
│  │                                                       │  │
│  │ Decision: EXCLUDE (changed)                          │  │
│  │ Reasoning: After reviewing, the abstract states     │  │
│  │ "adolescents aged 13-17", which falls outside our   │  │
│  │ 5-12 year range. Changing to EXCLUDE.              │  │
│  │ Confidence: 0.90                                     │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ 🔍 Evaluator Final Assessment           Round 2      │  │
│  │                                                       │  │
│  │ Result: AGREE ✓                                      │  │
│  │ Reasoning: Correct decision after reconsideration.  │  │
│  │ Consensus Reached!                                   │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
│  Final Decision: EXCLUDE                                   │
│  Confidence: 0.90                                          │
│  Rounds: 2                                                 │
│  Cost: HKD $0.008                                         │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

**Real-Time Updates:**
- Poll every 2 seconds during evaluation
- Smooth scroll to latest round
- Highlight current active round
- Show typing indicator during LLM processing

### 4.3 Results Visualization

**Dashboard Metrics:**

```
┌──────────────────────────────────────────────────────┐
│  Evaluation Summary                                   │
├──────────────────────────────────────────────────────┤
│                                                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
│  │   Total     │ │   Agreed    │ │  Discussed  │   │
│  │    500      │ │     425     │ │      50     │   │
│  │  Evaluated  │ │    (85%)    │ │    (10%)    │   │
│  └─────────────┘ └─────────────┘ └─────────────┘   │
│                                                       │
│  ┌─────────────┐ ┌─────────────┐                    │
│  │   Human     │ │  Total Cost │                    │
│  │   Review    │ │  HKD $2.50  │                    │
│  │     25      │ │             │                    │
│  │    (5%)     │ │             │                    │
│  └─────────────┘ └─────────────┘                    │
│                                                       │
└──────────────────────────────────────────────────────┘
```

**Consensus Chart:**

```
Discussion Round Distribution
│
│ ████████████████████ 425 (85%) - Agreed Round 1
│ ████ 40 (8%) - Consensus Round 2
│ ██ 10 (2%) - Consensus Round 3
│ ██ 25 (5%) - No Consensus (Human Review)
│
└─────────────────────────────────────────────
```

**Quality Score Distribution:**

```
Rubric Score Histogram
│
│     ██
│     ██  ████
│  ██ ██  ████  ████
│  ██ ██  ████  ████  ██
│  ██ ██  ████  ████  ██
└──────────────────────────
  0-1 1-2  2-3   3-4  4-5
  
Average: 3.8/5.0
```

### 4.4 Manual Review Interface

**Human Review Queue:**

```
┌────────────────────────────────────────────────────────────┐
│  🚨 Items Flagged for Human Review (25 articles)           │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  Filter: [ All | No Consensus | Low Quality | Ambiguous ] │
│  Sort: [ Oldest First | Lowest Confidence | Most Rounds ]  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ Article #342                              Priority: 🔴 │
│  │ "Insulin pump therapy in young children..."         │  │
│  │                                                       │  │
│  │ Issue: No consensus after 3 rounds                  │  │
│  │ LLM Disagreement:                                    │  │
│  │  - Primary: INCLUDE (conf: 0.72)                    │  │
│  │  - Evaluator: Should EXCLUDE (age not specified)    │  │
│  │                                                       │  │
│  │ [View Discussion] [Review Now]                      │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ Article #127                              Priority: 🟡 │
│  │ "Meta-analysis of glycemic control..."              │  │
│  │                                                       │  │
│  │ Issue: Abstract quality too poor (Rubric: 2.1/5)    │  │
│  │ Missing: Specific outcomes, population details      │  │
│  │                                                       │  │
│  │ [View Discussion] [Review Now]                      │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

**Review Interface:**

```
┌────────────────────────────────────────────────────────────┐
│  Manual Review: Article #342                               │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  Title: Insulin pump therapy in young children with T1D   │
│  Abstract: [Full abstract text displayed]                  │
│                                                             │
│  Criteria:                                                  │
│  ✓ Population: Children 5-12 years with T1D               │
│  ✓ Intervention: Insulin therapy, glucose monitoring      │
│  ✓ Outcome: Glycemic control (HbA1c)                      │
│                                                             │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                                                             │
│  LLM Discussion Summary:                                    │
│  • Primary decided: INCLUDE (reasoning: matches criteria)  │
│  • Evaluator concern: Age range ambiguous in abstract      │
│  • Round 2: Primary defended decision                      │
│  • Round 3: Still no agreement                             │
│                                                             │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                                                             │
│  Your Decision:                                             │
│  ○ INCLUDE      ○ EXCLUDE      ○ UNCERTAIN                │
│                                                             │
│  Reasoning: [Text area for your notes]                     │
│                                                             │
│  Confidence: [Slider: 0% ──●──────── 100%]                │
│                                                             │
│  [Save Decision]  [Skip]  [Mark for Full-Text Review]     │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

---

## 🧪 Section 5: Testing & Deployment Strategy

### 5.1 Testing Approach

**Unit Testing:**

- **Data Models:** Validate Pydantic schemas, test serialization/deserialization
- **Database Operations:** Test CRUD operations for evaluation tables
- **Prompt Construction:** Verify prompt templates generate correct format
- **Rubric Calculations:** Test weighted scoring with edge cases
- **Consensus Detection:** Test agreement calculation with various scenarios

**Integration Testing:**

- **Single Evaluation:** Primary → Evaluator → Result (no discussion)
- **Discussion Loop:** Test full 3-round workflow
- **Human Review Trigger:** Verify HUMAN_NEEDED path works
- **Cost Tracking:** Validate token counting and cost accumulation
- **State Persistence:** Ensure discussion rounds save correctly

**End-to-End Testing:**

```
Test Workflow:
1. Run AI screening on 100 test articles
2. Run evaluation on all 100 results
3. Verify outcomes:
   - ~70% should AGREE immediately
   - ~20% should enter discussion (1-2 rounds)
   - ~10% should flag for human review
4. Check database records
5. Verify frontend displays correctly
6. Test manual review interface
```

**Quality Validation:**

- **Baseline Comparison:** Compare dual-LLM vs single-LLM on 500 manually-reviewed articles
- **Inter-Rater Reliability:** Measure agreement with human reviewers
- **False Positive/Negative Rates:** Track screening accuracy improvements
- **Discussion Effectiveness:** Measure how often discussion changes incorrect decisions

### 5.2 Performance Benchmarks

**Target Metrics:**

| Metric | Target | Measurement |
|--------|--------|-------------|
| Evaluation Speed | <5 seconds per article | Time from API call to result |
| Discussion Round Time | <8 seconds per round | LLM response latency |
| Agreement Rate (Round 1) | 70-80% | % that agree immediately |
| Consensus Rate (Round 2-3) | 15-20% | % reaching consensus in discussion |
| Human Review Rate | 5-10% | % flagged for manual review |
| Cost per Article (Evaluated) | <HKD $0.10 | Total API cost (primary + eval) |
| Database Write Time | <100ms | Save evaluation session |

**Load Testing:**

- Test 1,000 concurrent evaluations
- Verify no database deadlocks
- Monitor memory usage (<2GB for 1,000 sessions)
- Check API rate limit handling

### 5.3 A/B Testing Framework

**Comparison Setup:**

```
Control Group (Single-LLM):
├─ 2,500 articles
├─ Standard AI screening only
├─ Track: decisions, confidence, cost, time
└─ Manual validation on random 100 sample

Treatment Group (Dual-LLM):
├─ 2,500 articles (same set)
├─ AI screening + Evaluation + Discussion
├─ Track: same metrics + discussion rounds
└─ Manual validation on same 100 sample
```

**Success Criteria:**

- **Accuracy Improvement:** ≥5% increase in agreement with human reviewers
- **False Positive Reduction:** ≥10% reduction vs single-LLM
- **Cost Efficiency:** Cost increase ≤15% for quality improvement
- **Time Acceptable:** Total time ≤2x single-LLM time

**Analysis Metrics:**

```
Quality Metrics:
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1 Score: 2 × (Precision × Recall) / (Precision + Recall)
- Cohen's Kappa: Inter-rater reliability with humans

Cost Metrics:
- Cost per correct decision
- Cost per prevented error
- ROI: (Errors prevented × value) - Extra cost

Efficiency Metrics:
- Human review reduction rate
- Time saved in manual screening
- Total workflow throughput
```

### 5.4 Deployment Considerations

**Rollout Strategy:**

**Phase 1: Internal Beta**
- Enable for development team only
- Test with historical datasets
- Collect feedback on UI/UX
- Tune prompts and thresholds

**Phase 2: Limited Release**
- Offer as optional feature
- Toggle: "Enable Dual-LLM Evaluation" in Settings
- Monitor usage patterns
- Gather cost/quality data

**Phase 3: Production**
- Default to dual-LLM for new projects
- Provide clear cost estimates upfront
- Allow mode selection per project
- Document best practices

**Infrastructure Requirements:**

- Database: Add indexes on `evaluation_sessions.screening_result_id`
- API: Handle 2-3x API call volume
- Storage: ~2MB per 1,000 evaluations (discussion history)
- Monitoring: Track evaluation success/failure rates

**Rollback Plan:**

If issues arise:
1. Disable evaluation for new tasks
2. Allow existing evaluations to complete
3. Revert to single-LLM mode
4. Preserve evaluation data for analysis

---

## 🔧 Section 6: Implementation Priorities & Dependencies

### 6.1 Build Order & Dependencies

**Phase 1: Foundation (Required First)**

```
┌─────────────────────────────────────┐
│ 1. Database Schema                  │ ← Start here
│    - Create evaluation tables       │
│    - Run migration script           │
│    - Test CRUD operations           │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 2. Data Models                      │
│    - Pydantic schemas               │
│    - API request/response models    │
│    - Validation rules               │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 3. Configuration                    │
│    - Evaluation config              │
│    - Rubric weights                 │
│    - Thresholds                     │
└─────────────────────────────────────┘
```

**Phase 2: Core Engine (Sequential)**

```
┌─────────────────────────────────────┐
│ 4. Single Evaluation Logic          │
│    - Evaluator LLM client           │
│    - Prompt construction            │
│    - Response parsing               │
│    - Rubric calculation             │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 5. Discussion Loop                  │
│    - Round management               │
│    - Consensus detection            │
│    - State tracking                 │
│    - Termination logic              │
└─────────────────────────────────────┘
```

**Phase 3: API & Integration (Parallel)**

```
┌─────────────────────────────────────┐  ┌─────────────────────────────────────┐
│ 6a. Backend API Endpoints           │  │ 6b. Frontend Components             │
│     - Start evaluation              │  │     - Evaluation page UI            │
│     - Get progress                  │  │     - Discussion viewer             │
│     - Get results                   │  │     - Manual review interface       │
│     - Manual review submit          │  │     - Charts & visualizations       │
└─────────────────────────────────────┘  └─────────────────────────────────────┘
             │                                          │
             └──────────────┬───────────────────────────┘
                            ▼
                   Integration Testing
```

**Phase 4: Polish & Deploy (Final)**

```
┌─────────────────────────────────────┐
│ 7. Testing & Optimization           │
│    - Unit tests                     │
│    - Integration tests              │
│    - Performance tuning             │
│    - A/B testing setup              │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 8. Documentation & Deployment       │
│    - User guide                     │
│    - API docs                       │
│    - Deployment scripts             │
│    - Monitoring setup               │
└─────────────────────────────────────┘
```

### 6.2 MVP vs Full Feature Set

**Minimum Viable Product (MVP):**

Essential features for first release:
- ✅ Database schema
- ✅ Single evaluation (no discussion)
- ✅ Basic rubric scoring
- ✅ Frontend: View evaluation results
- ✅ Manual review for flagged items
- ❌ Discussion loop (can add later)
- ❌ Advanced visualizations

**MVP Timeline:** 2-3 weeks

**Full Feature Set:**

Complete implementation:
- ✅ All MVP features
- ✅ Discussion loop (3 rounds)
- ✅ Consensus detection
- ✅ Real-time discussion viewer
- ✅ Advanced charts & analytics
- ✅ A/B testing framework
- ✅ Cost optimization

**Full Timeline:** 6-8 weeks

### 6.3 Parallel Development Opportunities

**Can Build Simultaneously:**

1. **Backend + Frontend**
   - Different developers can work independently
   - Use mock API responses for frontend dev
   - Integrate when both ready

2. **Core Logic + Database**
   - Database team sets up schema
   - Logic team develops algorithms
   - Integrate via ORM layer

3. **Evaluation Engine + Discussion Loop**
   - Build single evaluation first
   - Add discussion as enhancement
   - Same LLM client, different workflow

**Must Build Sequentially:**

1. Database → Models → API
2. Single Evaluation → Discussion Loop
3. Backend API → Frontend Integration

### 6.4 Risk Mitigation

**Technical Risks:**

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM API downtime | High | Implement retry logic, queue failed evaluations |
| Discussion loops don't converge | Medium | Max 3 rounds, force decision or flag |
| Database deadlocks | Medium | Use proper locking, test concurrency |
| High API costs | High | Cost limits, selective evaluation mode |

**Quality Risks:**

| Risk | Impact | Mitigation |
|------|--------|------------|
| Evaluator LLM too harsh/lenient | High | Tune prompts, adjust thresholds based on A/B test |
| Discussion doesn't improve quality | High | Measure with A/B test, disable if ineffective |
| Too many human reviews flagged | Medium | Adjust HUMAN_NEEDED thresholds |

---

## 💰 Section 7: Cost-Benefit Analysis

### 7.1 Cost Breakdown

**Single-LLM Screening (Current System):**

```
Per Article Cost:
├─ Primary LLM screening: HKD $0.05
├─ Total: HKD $0.05

For 5,000 articles:
└─ Total Cost: HKD $250
```

**Dual-LLM with Full Evaluation:**

```
Per Article Cost:
├─ Primary LLM screening: HKD $0.05
├─ Evaluator LLM (single pass): HKD $0.004
├─ Discussion (avg 0.5 rounds): HKD $0.006
├─ Total: HKD $0.060

For 5,000 articles:
└─ Total Cost: HKD $300 (+20% overhead)
```

**Selective Evaluation (Optimized):**

```
Evaluation Strategy:
├─ High confidence (>0.8): No evaluation (3,000 articles)
├─ Medium confidence (0.6-0.8): Full evaluation (1,500 articles)
├─ Low confidence (<0.6): Auto-flag for human (500 articles)

Cost Calculation:
├─ All screening: 5,000 × $0.05 = $250
├─ Evaluations: 1,500 × $0.010 = $15
├─ Total: HKD $265 (+6% overhead)
```

### 7.2 Expected Quality Improvements

**Baseline (Single-LLM):**

From existing data:
- Accuracy vs human: 88%
- False positives: 8%
- False negatives: 4%
- Human review needed: 15%

**Projected (Dual-LLM):**

Based on similar systems:
- Accuracy vs human: 93% (+5%)
- False positives: 4% (-50%)
- False negatives: 3% (-25%)
- Human review needed: 8% (-47%)

**Impact on Workflow:**

```
5,000 Articles Screened:

Single-LLM:
├─ Errors: 600 (12%)
├─ Manual review: 750 (15%)
├─ Researcher time: ~40 hours
└─ Cost: HKD $250

Dual-LLM (Full):
├─ Errors: 350 (7%) - 42% reduction
├─ Manual review: 400 (8%) - 47% reduction
├─ Researcher time: ~22 hours - 45% reduction
└─ Cost: HKD $300 (+$50)

ROI: 18 hours saved × HKD $300/hour = $5,400 saved
     Cost increase: $50
     Net benefit: $5,350 🎉
```

### 7.3 When to Use Dual-LLM vs Single-LLM

**Use Dual-LLM Evaluation When:**

✅ **High-stakes reviews**
   - Systematic reviews for policy/guidelines
   - Meta-analyses for clinical decisions
   - Grant-funded research with strict quality requirements

✅ **Large datasets (>1,000 articles)**
   - Cost overhead is small percentage
   - Time savings are substantial
   - Human review reduction valuable

✅ **Ambiguous criteria**
   - Inclusion/exclusion rules are complex
   - Subjective judgments required
   - Need quality assurance

✅ **Budget allows 10-20% cost increase**
   - Research funding covers AI costs
   - Value time savings over direct costs

**Use Single-LLM Screening When:**

❌ **Pilot studies or exploratory reviews**
   - Initial scoping to estimate scope
   - Rough filtering before detailed review

❌ **Very clear criteria**
   - Binary inclusion rules (e.g., "only RCTs")
   - Objective filters (e.g., "published after 2020")

❌ **Small datasets (<500 articles)**
   - Manual review is already fast
   - Marginal time savings

❌ **Tight budget**
   - Every dollar counts
   - Can accept more manual review

### 7.4 Cost Optimization Strategies

**Strategy 1: Selective Evaluation**

Only evaluate borderline decisions:

```python
if primary_confidence >= 0.8:
    # High confidence - skip evaluation
    accept_decision()
elif primary_confidence >= 0.6:
    # Medium confidence - evaluate
    run_evaluation()
else:
    # Low confidence - flag for human
    flag_for_review()
```

**Savings:** 60% reduction in evaluation costs

**Strategy 2: Prompt Caching**

Cache system prompt and criteria across evaluations:

```python
# First evaluation
cached_prompt = cache_prompt(system_prompt + criteria)
# Cost: Full input tokens

# Subsequent evaluations (same criteria)
use_cached_prompt(cached_prompt)
# Cost: Only article-specific tokens (~70% reduction)
```

**Savings:** 30% reduction in input token costs

**Strategy 3: Batch Processing**

Process multiple evaluations in parallel:

```python
# Instead of sequential:
for article in articles:
    evaluate(article)  # ~5 seconds each

# Parallel with 10 workers:
with ThreadPoolExecutor(max_workers=10):
    results = executor.map(evaluate, articles)
# ~5 seconds total for 10 articles
```

**Savings:** No cost savings, but 10x faster wall-clock time

**Strategy 4: Smart Discussion Limits**

Limit discussion rounds based on confidence gap:

```python
if abs(primary_confidence - evaluator_confidence) < 0.2:
    max_rounds = 1  # Close confidence - quick discussion
elif abs(primary_confidence - evaluator_confidence) < 0.4:
    max_rounds = 2  # Moderate gap
else:
    max_rounds = 3  # Large disagreement - full discussion
```

**Savings:** ~30% reduction in discussion costs

**Combined Optimization:**

```
Baseline Dual-LLM: HKD $300 for 5,000 articles

With optimizations:
├─ Selective evaluation: $265
├─ Prompt caching: $230
├─ Smart discussion limits: $210
└─ Final cost: HKD $210 (-30% from baseline)

Overhead vs Single-LLM: 
$210 vs $250 = -16% (Actually cheaper due to caching!)
```

### 7.5 Success Metrics

**Track These Metrics to Validate ROI:**

1. **Quality Metrics:**
   - Agreement with human reviewers (target: ≥93%)
   - Inter-rater reliability (Cohen's Kappa)
   - False positive/negative rates

2. **Efficiency Metrics:**
   - Human review reduction (target: ≥40%)
   - Time to complete screening
   - Researcher hours saved

3. **Cost Metrics:**
   - Cost per article
   - Cost per correct decision
   - Cost per error prevented

4. **User Satisfaction:**
   - Confidence in AI decisions
   - Willingness to use dual-LLM
   - Perceived value for money

**Decision Framework:**

```
If quality_improvement ≥5% AND cost_increase ≤20%:
    → Dual-LLM is worth it
elif time_saved_hours × hourly_rate > cost_increase:
    → Dual-LLM is worth it
elif human_review_reduction ≥40%:
    → Dual-LLM is worth it
else:
    → Stick with single-LLM
```

---

## ✅ Implementation Complete

All sections of the Dual-LLM Deliberative Evaluation System implementation plan are now documented. This plan provides:

1. ✅ **System Architecture & Data Models** - Database schema, data models, configuration
2. ✅ **Core Evaluation Engine** - LLM prompts, rubric scoring, single-pass evaluation
3. ✅ **Discussion Loop Mechanism** - Iterative dialogue, consensus detection, termination logic
4. ✅ **Frontend Evaluation Page** - UI design, real-time viewer, manual review interface
5. ✅ **Testing & Deployment** - Testing strategy, benchmarks, A/B testing, rollout plan
6. ✅ **Implementation Priorities** - Build order, MVP scope, parallel development
7. ✅ **Cost-Benefit Analysis** - Cost breakdown, quality improvements, optimization strategies

**Next Steps:**
1. Review and approve this plan
2. Prioritize MVP vs full feature set
3. Allocate development resources
4. Begin Phase 1: Database schema implementation

---

*Plan completed: November 11, 2025*
