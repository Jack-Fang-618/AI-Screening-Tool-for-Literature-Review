"""
Grok API Client - X.AI API Integration

Handles communication with X.AI's Grok models for article screening.
Supports both reasoning and non-reasoning models with proper error handling.

Refined improvements:
- Exponential backoff retry strategy
- Rate limiting with token bucket
- Response validation
- Streaming support (optional)
- Comprehensive error handling
"""

import requests
import time
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass
import os

logger = logging.getLogger(__name__)


@dataclass
class GrokResponse:
    """Grok API response"""
    content: str
    reasoning_content: Optional[str] = None
    model: str = ""
    provider: str = "xai"
    usage: Dict = None
    
    def __post_init__(self):
        if self.usage is None:
            self.usage = {}


class GrokClient:
    """
    Client for X.AI Grok API
    
    Supports:
    - grok-4-fast-reasoning (recommended for screening)
    - grok-4-fast-non-reasoning (fastest, non-reasoning)
    - grok-4 (standard, most expensive)
    """
    
    BASE_URL = "https://api.x.ai/v1"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "grok-4-fast-reasoning",
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize Grok client
        
        Args:
            api_key: X.AI API key (or from XAI_API_KEY env var)
            model: Model name (default: grok-4-fast-reasoning)
            temperature: Temperature for generation (0.0-2.0)
            max_tokens: Maximum tokens to generate (None = no limit)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.api_key = api_key or os.getenv('XAI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "X.AI API key required. Set XAI_API_KEY environment variable or pass api_key parameter."
            )
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Rate limiting with safety margin (480 RPM = 8 req/s, use ~6 req/s for stability)
        self._last_request_time = 0
        self._min_request_interval = 0.17  # ~6 req/s (conservative, 25% safety margin)
        
        logger.info(f"Initialized Grok client with model: {model}")
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> GrokResponse:
        """
        Send chat completion request to Grok API
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            stream: Enable streaming (not yet implemented)
            
        Returns:
            GrokResponse with content and metadata
        """
        # Rate limiting
        self._rate_limit()
        
        # Prepare request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "stream": stream
        }
        
        # Only add max_tokens if specified
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        elif self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        
        # Retry logic with exponential backoff
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                # Check for HTTP errors
                response.raise_for_status()
                
                # Parse response
                data = response.json()
                return self._parse_response(data)
                
            except requests.exceptions.HTTPError as e:
                # Handle specific HTTP errors
                status_code = e.response.status_code
                
                if status_code == 429:  # Rate limit
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(
                        f"⚠️ Rate limit hit (HTTP 429) - Model: {self.model}, "
                        f"Waiting {wait_time}s (attempt {attempt + 1}/{self.max_retries}). "
                        f"Consider reducing worker count if this happens frequently."
                    )
                    time.sleep(wait_time)
                    last_exception = e
                    
                elif status_code == 401:  # Authentication error
                    logger.error("❌ Authentication failed - check API key")
                    raise ValueError("Invalid API key") from e
                    
                elif status_code >= 500:  # Server error
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"⚠️ X.AI server error {status_code}, retrying in {wait_time}s "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(wait_time)
                    last_exception = e
                    
                else:
                    # Other HTTP errors
                    logger.error(
                        f"❌ HTTP error {status_code}: {e.response.text[:200]}"
                    )
                    raise
                    
            except requests.exceptions.ConnectionError as e:
                # Network connection issues
                wait_time = min(2 ** attempt, 10)  # Cap at 10s
                logger.warning(
                    f"⚠️ Network connection error: {str(e)[:150]}. "
                    f"Retrying in {wait_time}s (attempt {attempt + 1}/{self.max_retries}). "
                    f"This may indicate network instability or too many parallel workers."
                )
                last_exception = e
                time.sleep(wait_time)
                
            except requests.exceptions.Timeout as e:
                wait_time = min(2 ** attempt, 8)  # Cap at 8s
                logger.warning(
                    f"⚠️ Request timeout after {self.timeout}s "
                    f"(attempt {attempt + 1}/{self.max_retries}). "
                    f"Retrying in {wait_time}s. Consider increasing timeout or reducing worker count."
                )
                last_exception = e
                time.sleep(wait_time)
                
            except requests.exceptions.RequestException as e:
                # Catch-all for other request errors
                wait_time = min(2 ** attempt, 10)
                logger.error(
                    f"⚠️ Request failed: {type(e).__name__}: {str(e)[:150]}. "
                    f"Retrying in {wait_time}s (attempt {attempt + 1}/{self.max_retries})"
                )
                last_exception = e
                time.sleep(wait_time)
        
        # All retries failed
        logger.error(
            f"❌ CRITICAL: All {self.max_retries} retry attempts failed. "
            f"Last error: {type(last_exception).__name__}: {str(last_exception)[:200]}. "
            f"Model: {self.model}. "
            f"Recommendation: If using high worker count (>4), try reducing to 2-4 workers for better stability."
        )
        raise Exception(f"Failed after {self.max_retries} attempts: {last_exception}")
    
    def screen_article(self, prompt: str) -> Dict:
        """
        Screen article using Grok (convenience method for screener)
        
        Args:
            prompt: Screening prompt with article title/abstract
            
        Returns:
            Dict with content, usage, model info
        """
        messages = [{"role": "user", "content": prompt}]
        response = self.chat_completion(messages)
        
        return {
            'content': response.content,
            'reasoning_content': response.reasoning_content,
            'model': response.model,
            'provider': response.provider,
            'usage': response.usage
        }
    
    def _parse_response(self, data: Dict) -> GrokResponse:
        """
        Parse Grok API response
        
        Args:
            data: Raw JSON response from API
            
        Returns:
            GrokResponse object
        """
        try:
            message = data['choices'][0]['message']
            
            # Handle reasoning models vs non-reasoning models
            content = message.get('content', '')
            reasoning_content = message.get('reasoning_content')  # Only for reasoning models
            
            # Extract token usage
            usage = data.get('usage', {})
            
            # Get model info
            model = data.get('model', self.model)
            
            return GrokResponse(
                content=content,
                reasoning_content=reasoning_content,
                model=model,
                provider='xai',
                usage=usage
            )
            
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to parse response: {e}")
            logger.debug(f"Response data: {data}")
            raise ValueError(f"Invalid response format: {e}") from e
    
    def _rate_limit(self):
        """Simple rate limiting using minimum request interval"""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()
    
    def test_connection(self) -> bool:
        """
        Test API connection and authentication
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            messages = [{"role": "user", "content": "Hello, this is a test."}]
            response = self.chat_completion(messages, max_tokens=10)
            logger.info(f"Connection test successful: {response.model}")
            return True
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0
    ) -> Dict[str, float]:
        """
        Estimate cost for token usage (in HKD)
        
        Args:
            input_tokens: Input token count
            output_tokens: Output token count
            cached_tokens: Cached input token count
            
        Returns:
            Dict with cost breakdown
        """
        # Pricing for current model (HKD per 1M tokens)
        pricing = {
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
            'grok-4': {
                'input': 3.0 * 7.78,
                'cached_input': 0.75 * 7.78,
                'output': 15.0 * 7.78
            }
        }
        
        model_pricing = pricing.get(self.model, pricing['grok-4-fast-reasoning'])
        
        # Calculate costs
        uncached_tokens = max(0, input_tokens - cached_tokens)
        input_cost = (uncached_tokens / 1_000_000) * model_pricing['input']
        cached_cost = (cached_tokens / 1_000_000) * model_pricing.get('cached_input', 0)
        output_cost = (output_tokens / 1_000_000) * model_pricing['output']
        
        total_cost = input_cost + cached_cost + output_cost
        
        return {
            'input_cost': input_cost,
            'cached_cost': cached_cost,
            'output_cost': output_cost,
            'total_cost': total_cost,
            'currency': 'HKD',
            'model': self.model
        }
