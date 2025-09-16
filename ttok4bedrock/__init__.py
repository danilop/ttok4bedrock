"""
ttok4bedrock - Token counting for Amazon Bedrock models
Drop-in replacement for ttok with Amazon Bedrock support
"""

from typing import Union, List, Optional
import boto3
from botocore.exceptions import ClientError
import json

__version__ = "0.1.0"
__all__ = ["count_tokens", "truncate"]


def count_tokens(
    text: str,
    model: str = "anthropic.claude-sonnet-4-20250514-v1:0",
    aws_region: Optional[str] = None
) -> int:
    """
    Count tokens for given text using Anthropic Claude models on Bedrock.
    
    Supports Anthropic Claude models with CountTokens API:
    - anthropic.claude-sonnet-4-20250514-v1:0 (default)
    - anthropic.claude-3-5-haiku-20241022-v1:0
    - anthropic.claude-3-5-sonnet-20241022-v2:0
    - anthropic.claude-3-5-sonnet-20240620-v1:0
    - anthropic.claude-3-7-sonnet-20250219-v1:0
    - anthropic.claude-opus-4-20250514-v1:0
    
    Args:
        text: Input text to count tokens for
        model: Full Bedrock model ID
        aws_region: AWS region (optional, uses default if not specified)
    
    Returns:
        Token count as integer
    """
    from .bedrock_counter import BedrockTokenCounter
    counter = BedrockTokenCounter(region=aws_region)
    return counter.count_tokens(text, model)


def truncate(
    text: str,
    max_tokens: int,
    model: str = "anthropic.claude-sonnet-4-20250514-v1:0",
    aws_region: Optional[str] = None
) -> str:
    """
    Truncate text to specified token count using Anthropic Claude models.
    
    Supports Anthropic Claude models with CountTokens API:
    - anthropic.claude-sonnet-4-20250514-v1:0 (default)
    - anthropic.claude-3-5-haiku-20241022-v1:0
    - anthropic.claude-3-5-sonnet-20241022-v2:0
    - anthropic.claude-3-5-sonnet-20240620-v1:0
    - anthropic.claude-3-7-sonnet-20250219-v1:0
    - anthropic.claude-opus-4-20250514-v1:0
    
    Args:
        text: Input text to truncate
        max_tokens: Maximum number of tokens
        model: Full Bedrock model ID
        aws_region: AWS region (optional)
    
    Returns:
        Truncated text string
    """
    from .bedrock_counter import BedrockTokenCounter
    counter = BedrockTokenCounter(region=aws_region)
    return counter.truncate(text, max_tokens, model)

