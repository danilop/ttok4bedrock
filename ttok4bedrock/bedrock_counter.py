"""
Core Bedrock token counting implementation.
"""

import boto3
from botocore.exceptions import ClientError
import json
from typing import Optional, Dict, Any
from functools import lru_cache


class BedrockTokenCounter:
    """Token counter using Amazon Bedrock CountTokens API."""
    
    # Models that support the CountTokens API
    SUPPORTED_MODELS = [
        "anthropic.claude-sonnet-4-20250514-v1:0",
        "anthropic.claude-3-5-haiku-20241022-v1:0",
        "anthropic.claude-3-5-sonnet-20241022-v2:0", 
        "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "anthropic.claude-3-7-sonnet-20250219-v1:0",
        "anthropic.claude-opus-4-20250514-v1:0"
    ]
    
    def __init__(self, region: Optional[str] = None, cache_size: int = 1000):
        """
        Initialize Bedrock token counter.
        
        Args:
            region: AWS region (uses default if not specified)
            cache_size: Maximum number of token counts to cache (LRU)
        """
        self.region = region
        self._client = None
        self._cache_size = cache_size
        # Create LRU-cached version of _count_tokens_impl
        self._count_tokens_cached = lru_cache(maxsize=cache_size)(self._count_tokens_impl)
    
    @property
    def client(self):
        """Lazy initialization of Bedrock client."""
        if self._client is None:
            client_kwargs = {'service_name': 'bedrock-runtime'}
            if self.region:
                client_kwargs['region_name'] = self.region
            self._client = boto3.client(**client_kwargs)
        return self._client
    
    def _get_message_overhead(self, model_id: str) -> int:
        """
        Calculate the token overhead of the message structure.
        Uses the same LRU cache as regular token counting.
        """
        # Use a minimal text to calculate overhead - this will be cached by LRU
        minimal_text = "A"
        raw_count = self._count_tokens_cached(minimal_text, model_id)
        # The overhead is the raw count minus 1 (for the "A" character)
        return raw_count - 1
    
    def _count_tokens_impl(self, text: str, model_id: str) -> int:
        """
        Internal implementation of token counting (without caching).
        This method is wrapped with lru_cache in __init__.
        """
        input_data = self._format_input_for_model(text, model_id)
        
        # Call Bedrock CountTokens API - let errors bubble up
        response = self.client.count_tokens(
            modelId=model_id,
            input=input_data
        )
        
        return response.get("inputTokens", 0)
    
    @classmethod
    def get_supported_models(cls) -> list[str]:
        """
        Get list of models that support the CountTokens API.
        
        Returns:
            List of supported model IDs
        """
        return cls.SUPPORTED_MODELS.copy()
    
    def _format_input_for_model(self, text: str, model_id: str) -> Dict[str, Any]:
        """
        Format input for Anthropic Claude models with CountTokens API support.
        
        Args:
            text: Input text
            model_id: Full Bedrock model ID
        
        Returns:
            Formatted input dictionary for CountTokens API
        """
        # Only support Anthropic Claude models that support CountTokens API
        if model_id in self.SUPPORTED_MODELS:
            return {
                "invokeModel": {
                    "body": json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "messages": [
                            {"role": "user", "content": text}
                        ],
                        "max_tokens": 1
                    })
                }
            }
        else:
            models_list = ", ".join(self.SUPPORTED_MODELS)
            raise ValueError(
                f"Model {model_id} is not supported. "
                f"Please use one of the supported Anthropic Claude models: {models_list}"
            )
    
    def count_tokens(self, text: str, model_id: str) -> int:
        """
        Count tokens using Bedrock CountTokens API with LRU caching.
        Returns only the tokens from the text content, excluding message structure overhead.
        
        Args:
            text: Input text to count tokens for
            model_id: Full Bedrock model ID
        
        Returns:
            Token count as integer (excluding message overhead)
        
        Raises:
            ClientError: If Bedrock API returns an error
        """
        raw_count = self._count_tokens_cached(text, model_id)
        overhead = self._get_message_overhead(model_id)
        return max(0, raw_count - overhead)
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache hit/miss statistics
        """
        cache_info = self._count_tokens_cached.cache_info()
        return {
            'hits': cache_info.hits,
            'misses': cache_info.misses,
            'maxsize': cache_info.maxsize,
            'currsize': cache_info.currsize,
            'hit_rate': cache_info.hits / (cache_info.hits + cache_info.misses) if (cache_info.hits + cache_info.misses) > 0 else 0.0
        }
    
    def truncate(self, text: str, max_tokens: int, model_id: str, return_metadata: bool = False):
        """
        Truncate text to max tokens using adaptive learning algorithm.
        
        Algorithm:
        1. Check if truncation needed (1 API call)
        2. Use adaptive estimation that learns from each API call
        3. Converge to exact result with minimal calls
        
        Args:
            text: Input text to truncate
            max_tokens: Maximum number of tokens
            model_id: Full Bedrock model ID
            return_metadata: If True, returns (result, metadata) tuple
        
        Returns:
            Truncated text string, or (text, metadata) if return_metadata=True
            Metadata includes: {'api_calls': int, 'final_token_count': int}
        
        Raises:
            ClientError: If Bedrock API returns an error
        """
        # Step 1: Check if truncation is needed (1 API call)
        # Use raw API count internally for truncation algorithm
        full_count_raw = self._count_tokens_cached(text, model_id)
        api_calls = 1
        overhead = self._get_message_overhead(model_id)
        full_count = max(0, full_count_raw - overhead)
        
        if full_count <= max_tokens:
            if return_metadata:
                return text, {'api_calls': api_calls, 'final_token_count': full_count}
            return text
        
        # Step 2: Smart initial estimation
        # We know: full_count tokens for len(text) characters
        # We want: max_tokens for target_chars characters
        
        # Calculate base ratio using raw counts for internal algorithm
        chars_per_token = len(text) / full_count_raw
        
        # Smart estimation: account for text characteristics
        # - More punctuation = more tokens per char
        # - More spaces = more tokens per char  
        # - Longer words = fewer tokens per char
        punctuation_ratio = sum(1 for c in text if c in '.,!?;:') / len(text)
        space_ratio = text.count(' ') / len(text)
        avg_word_length = len(text.replace(' ', '')) / max(text.count(' ') + 1, 1)
        
        # Adjust ratio based on text characteristics
        adjustment_factor = 1.0
        if punctuation_ratio > 0.05:  # High punctuation
            adjustment_factor *= 0.95  # Slightly fewer chars per token
        if space_ratio > 0.15:  # High word density
            adjustment_factor *= 0.98  # Slightly fewer chars per token
        if avg_word_length > 8:  # Long words
            adjustment_factor *= 1.02  # Slightly more chars per token
            
        # Apply smart adjustment
        smart_chars_per_token = chars_per_token * adjustment_factor
        target_chars = int(max_tokens * smart_chars_per_token)
        target_chars = min(target_chars, len(text))
        
        best_text = ""
        best_count = 0
        
        # Adaptive learning loop - each call improves our estimation
        while api_calls < 20:  # Reasonable limit
            # Test current estimation
            test_text = text[:target_chars]
            
            try:
                count = self.count_tokens(test_text, model_id)
                api_calls += 1
                
                if count == max_tokens:
                    # Perfect match!
                    best_text = test_text
                    best_count = count
                    break
                elif count < max_tokens:
                    # We can add more characters
                    best_text = test_text
                    best_count = count
                    
                    # Calculate how many more chars we can add
                    remaining_tokens = max_tokens - count
                    if remaining_tokens > 0:
                        # Estimate additional chars needed
                        additional_chars = int(remaining_tokens * chars_per_token)
                        target_chars = min(target_chars + additional_chars, len(text))
                        
                        # If we're close, try adding one character at a time to find exact match
                        if additional_chars <= 5:  # Increased threshold
                            for i in range(len(best_text), min(len(best_text) + 15, len(text))):  # Increased search range
                                test_text = text[:i + 1]
                                try:
                                    count = self.count_tokens(test_text, model_id)
                                    api_calls += 1
                                    
                                    if count <= max_tokens:
                                        best_text = test_text
                                        best_count = count
                                        # If we found exact match, we're done
                                        if count == max_tokens:
                                            break
                                    else:
                                        break
                                except ClientError:
                                    break
                            break
                else:
                    # We have too many tokens, need to reduce
                    # Calculate how many chars to remove
                    excess_tokens = count - max_tokens
                    chars_to_remove = int(excess_tokens * chars_per_token)
                    target_chars = max(target_chars - chars_to_remove, 1)  # Don't go below 1
                    
                    # If we're close, try removing one character at a time
                    if chars_to_remove <= 5:  # Increased threshold
                        for i in range(len(best_text), 0, -1):
                            test_text = text[:i]
                            try:
                                count = self.count_tokens(test_text, model_id)
                                api_calls += 1
                                
                                if count <= max_tokens:
                                    best_text = test_text
                                    best_count = count
                                    # If we found exact match, we're done
                                    if count == max_tokens:
                                        break
                                    else:
                                        break
                            except ClientError:
                                break
                        break
                        
            except ClientError:
                # If we get an error, try a smaller text
                target_chars = max(target_chars - 10, 1)
                continue
        
        # Final verification and correction
        if best_text:
            final_token_count = self.count_tokens(best_text, model_id)
            api_calls += 1
            
            # If we're not exactly at the target, do an exhaustive search
            if final_token_count != max_tokens and api_calls < 20:
                # Try every possible length to find exact match
                for i in range(1, len(text) + 1):
                    if api_calls >= 20:
                        break
                        
                    test_text = text[:i]
                    test_count = self.count_tokens(test_text, model_id)
                    api_calls += 1
                    
                    if test_count == max_tokens:
                        # Found exact match!
                        best_text = test_text
                        final_token_count = test_count
                        break
                    elif test_count < max_tokens:
                        # Keep track of the best under-target result
                        best_text = test_text
                        final_token_count = test_count
                    # If test_count > max_tokens, we've gone too far, but keep searching
                    # in case there's an exact match later (due to irregular tokenization)
        else:
            final_token_count = 0
        
        if return_metadata:
            return best_text, {'api_calls': api_calls, 'final_token_count': final_token_count}
        return best_text