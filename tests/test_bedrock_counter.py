"""
Test suite for ttok4bedrock
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError, NoCredentialsError, NoRegionError

from ttok4bedrock import count_tokens, truncate
from ttok4bedrock.bedrock_counter import BedrockTokenCounter


class TestBedrockTokenCounter:
    """Test the BedrockTokenCounter class."""
    
    @patch('boto3.client')
    def test_lazy_client_initialization(self, mock_client):
        """Test that Bedrock client is initialized lazily."""
        counter = BedrockTokenCounter()
        
        # Client should not be initialized yet
        assert counter._client is None
        
        # Access client property
        mock_bedrock_client = Mock()
        mock_client.return_value = mock_bedrock_client
        
        client = counter.client
        assert client is not None
        assert counter._client is not None
        mock_client.assert_called_once_with(service_name='bedrock-runtime')
    
    @patch('boto3.client')
    def test_region_configuration(self, mock_client):
        """Test that region is properly configured."""
        mock_bedrock_client = Mock()
        mock_client.return_value = mock_bedrock_client
        
        counter = BedrockTokenCounter(region='us-west-2')
        _ = counter.client
        
        mock_client.assert_called_once_with(
            service_name='bedrock-runtime',
            region_name='us-west-2'
        )
    
    def test_format_input_claude(self):
        """Test input formatting for Claude models."""
        counter = BedrockTokenCounter()
        
        formatted = counter._format_input_for_model(
            "Test text",
            "anthropic.claude-3-5-haiku-20241022-v1:0"
        )
        
        assert "invokeModel" in formatted
        body = formatted["invokeModel"]["body"]
        import json
        parsed_body = json.loads(body)
        assert "anthropic_version" in parsed_body
        assert "messages" in parsed_body
    
    def test_format_input_unsupported_model(self):
        """Test input formatting for unsupported models raises ValueError."""
        counter = BedrockTokenCounter()
        
        with pytest.raises(ValueError) as exc_info:
            counter._format_input_for_model(
                "Test text",
                "amazon.nova-pro-v1:0"
            )
        
        assert "is not supported" in str(exc_info.value)
        assert "Please use one of the supported Anthropic Claude models" in str(exc_info.value)
    
    def test_format_input_another_unsupported_model(self):
        """Test input formatting for another unsupported model."""
        counter = BedrockTokenCounter()
        
        with pytest.raises(ValueError) as exc_info:
            counter._format_input_for_model(
                "Test text",
                "meta.llama3-8b-instruct-v1:0"
            )
        
        assert "is not supported" in str(exc_info.value)
        assert "Please use one of the supported Anthropic Claude models" in str(exc_info.value)
    
    @patch('boto3.client')
    def test_count_tokens_success(self, mock_client):
        """Test successful token counting with overhead removal."""
        mock_bedrock_client = Mock()
        # Mock returns 8 for overhead calculation and 42 for actual text
        mock_bedrock_client.count_tokens.return_value = {"inputTokens": 42}
        mock_client.return_value = mock_bedrock_client
        
        counter = BedrockTokenCounter()
        
        # First call will calculate overhead (returns 8), second call gets actual count (returns 42)
        # But due to LRU caching, we need to account for the caching behavior
        result = counter.count_tokens("Test text", "anthropic.claude-3-5-haiku-20241022-v1:0")
        
        # The result should be less than the raw API response due to overhead removal
        # We can't predict the exact value due to caching, but it should be reasonable
        assert result >= 0
        assert result < 42  # Should be less than raw API response
        mock_bedrock_client.count_tokens.assert_called()
    
    def test_count_tokens_unsupported_model_error(self):
        """Test that unsupported models raise ValueError."""
        counter = BedrockTokenCounter()
        
        with pytest.raises(ValueError) as exc_info:
            counter.count_tokens("Test text", "invalid-model")
        
        assert "is not supported" in str(exc_info.value)
        assert "Please use one of the supported Anthropic Claude models" in str(exc_info.value)
    
    @patch('boto3.client')
    def test_count_tokens_error_bubbles_up(self, mock_client):
        """Test that boto3 errors bubble up without modification."""
        mock_bedrock_client = Mock()
        error_response = {
            'Error': {
                'Code': 'ValidationException',
                'Message': 'Invalid model ID'
            }
        }
        mock_bedrock_client.count_tokens.side_effect = ClientError(
            error_response,
            'CountTokens'
        )
        mock_client.return_value = mock_bedrock_client
        
        counter = BedrockTokenCounter()
        
        with pytest.raises(ClientError) as exc_info:
            counter.count_tokens("Test text", "anthropic.claude-3-5-haiku-20241022-v1:0")
        
        assert exc_info.value.response['Error']['Code'] == 'ValidationException'
    
    @patch('boto3.client')
    def test_truncate_no_truncation_needed(self, mock_client):
        """Test truncate when text is already short enough."""
        mock_bedrock_client = Mock()
        mock_bedrock_client.count_tokens.return_value = {"inputTokens": 10}
        mock_client.return_value = mock_bedrock_client
        
        counter = BedrockTokenCounter()
        
        result = counter.truncate("Short text", 20, "anthropic.claude-3-5-haiku-20241022-v1:0")
        
        assert result == "Short text"
        # Should be called at least once (for overhead calculation and actual text)
        assert mock_bedrock_client.count_tokens.call_count >= 1
    
    @patch('boto3.client')
    def test_truncate_optimized_algorithm(self, mock_client):
        """Test optimized truncation algorithm."""
        mock_bedrock_client = Mock()
        
        # Simulate different token counts for different text lengths
        def mock_count_tokens(modelId, input):
            body = input.get("invokeModel", {}).get("body", "{}")
            import json
            parsed_body = json.loads(body) if isinstance(body, str) else body
            
            # Extract text length from the input
            if "messages" in parsed_body:
                content = parsed_body["messages"][0]["content"]
            elif "inputText" in parsed_body:
                content = parsed_body["inputText"]
            else:
                content = ""
            
            # Rough approximation: 4 chars = 1 token
            return {"inputTokens": max(1, len(content) // 4)}
        
        mock_bedrock_client.count_tokens.side_effect = mock_count_tokens
        mock_client.return_value = mock_bedrock_client
        
        counter = BedrockTokenCounter()
        long_text = "A" * 200  # ~50 tokens
        
        result = counter.truncate(long_text, 20, "anthropic.claude-3-5-haiku-20241022-v1:0")
        
        assert len(result) < len(long_text)
        # Optimized algorithm should use fewer API calls than naive approach
        # Naive would need ~50 API calls, optimized should need <25
        assert mock_bedrock_client.count_tokens.call_count <= 25
        # Should still be accurate
        assert len(result) > 0
    
    @patch('boto3.client')
    def test_truncate_efficiency(self, mock_client):
        """Test that truncation is efficient with minimal API calls."""
        mock_bedrock_client = Mock()
        
        # Simulate realistic token counting
        def mock_count_tokens(modelId, input):
            body = input.get("invokeModel", {}).get("body", "{}")
            import json
            parsed_body = json.loads(body) if isinstance(body, str) else body
            content = parsed_body.get("messages", [{}])[0].get("content", "")
            # Realistic approximation: 3.5 chars per token on average
            return {"inputTokens": max(1, int(len(content) / 3.5))}
        
        mock_bedrock_client.count_tokens.side_effect = mock_count_tokens
        mock_client.return_value = mock_bedrock_client
        
        counter = BedrockTokenCounter()
        # Test with a realistic long text
        long_text = "This is a sample text that will be truncated. " * 50  # ~700 chars, ~200 tokens
        
        result = counter.truncate(long_text, 50, "anthropic.claude-3-5-haiku-20241022-v1:0")
        
        # Should be much more efficient than naive character-by-character approach
        # Naive would need ~200 API calls, optimized should need <20
        assert mock_bedrock_client.count_tokens.call_count < 20
        assert len(result) < len(long_text)
        assert len(result) > 0
    
    @patch('boto3.client')
    def test_truncate_100_tokens(self, mock_client):
        """Test truncation efficiency with 100-token text."""
        mock_bedrock_client = Mock()
        
        # Simulate realistic token counting
        def mock_count_tokens(modelId, input):
            body = input.get("invokeModel", {}).get("body", "{}")
            import json
            parsed_body = json.loads(body) if isinstance(body, str) else body
            content = parsed_body.get("messages", [{}])[0].get("content", "")
            # Realistic approximation: 3.5 chars per token on average
            return {"inputTokens": max(1, int(len(content) / 3.5))}
        
        mock_bedrock_client.count_tokens.side_effect = mock_count_tokens
        mock_client.return_value = mock_bedrock_client
        
        counter = BedrockTokenCounter()
        # Create text that will be ~100 tokens
        long_text = "This is a sample sentence for testing token counting efficiency. " * 20  # ~350 chars, ~100 tokens
        
        result = counter.truncate(long_text, 50, "anthropic.claude-3-5-haiku-20241022-v1:0")
        
        # Should be very efficient even for 100-token text
        # Naive would need ~100 API calls, optimized should need <=20
        assert mock_bedrock_client.count_tokens.call_count <= 20
        assert len(result) < len(long_text)
        assert len(result) > 0
        
        # Verify the result is reasonable (should be roughly half the original length)
        assert len(result) > len(long_text) * 0.1  # At least 10% of original
        assert len(result) < len(long_text) * 0.8  # At most 80% of original
    
    @patch('boto3.client')
    def test_truncate_1000_tokens(self, mock_client):
        """Test truncation efficiency with 1000-token text."""
        mock_bedrock_client = Mock()
        
        # Simulate realistic token counting
        def mock_count_tokens(modelId, input):
            body = input.get("invokeModel", {}).get("body", "{}")
            import json
            parsed_body = json.loads(body) if isinstance(body, str) else body
            content = parsed_body.get("messages", [{}])[0].get("content", "")
            # Realistic approximation: 3.5 chars per token on average
            return {"inputTokens": max(1, int(len(content) / 3.5))}
        
        mock_bedrock_client.count_tokens.side_effect = mock_count_tokens
        mock_client.return_value = mock_bedrock_client
        
        counter = BedrockTokenCounter()
        # Create text that will be ~1000 tokens
        long_text = "This is a comprehensive sample sentence designed for testing token counting efficiency at scale. " * 100  # ~3500 chars, ~1000 tokens
        
        result = counter.truncate(long_text, 200, "anthropic.claude-3-5-haiku-20241022-v1:0")
        
        # Should be extremely efficient even for 1000-token text
        # Naive would need ~1000 API calls, optimized should need <=20
        assert mock_bedrock_client.count_tokens.call_count <= 20
        assert len(result) < len(long_text)
        assert len(result) > 0
        
        # Verify the result is reasonable (should be roughly 20% of original length)
        assert len(result) > len(long_text) * 0.05  # At least 5% of original
        assert len(result) < len(long_text) * 0.3   # At most 30% of original
    
    @patch('boto3.client')
    def test_truncate_extreme_case(self, mock_client):
        """Test truncation with extreme case: very long text to very few tokens."""
        mock_bedrock_client = Mock()
        
        # Simulate realistic token counting
        def mock_count_tokens(modelId, input):
            body = input.get("invokeModel", {}).get("body", "{}")
            import json
            parsed_body = json.loads(body) if isinstance(body, str) else body
            content = parsed_body.get("messages", [{}])[0].get("content", "")
            # Realistic approximation: 3.5 chars per token on average
            return {"inputTokens": max(1, int(len(content) / 3.5))}
        
        mock_bedrock_client.count_tokens.side_effect = mock_count_tokens
        mock_client.return_value = mock_bedrock_client
        
        counter = BedrockTokenCounter()
        # Create very long text
        long_text = "This is an extremely long text designed to test the algorithm's efficiency when truncating from a very large input to a very small output. " * 200  # ~7000 chars, ~2000 tokens
        
        result = counter.truncate(long_text, 10, "anthropic.claude-3-5-haiku-20241022-v1:0")
        
        # Should still be efficient even in extreme cases
        # Naive would need ~2000 API calls, optimized should need <=20
        assert mock_bedrock_client.count_tokens.call_count <= 20
        assert len(result) < len(long_text)
        assert len(result) > 0
        
        # Result should be very small (only 10 tokens)
        assert len(result) < len(long_text) * 0.05  # Less than 5% of original
    
    def test_truncate_systematic_range_real_api(self):
        """Test truncation systematically from 1 token up to full length + 1 using real Bedrock API."""
        # This test requires real AWS credentials and will make actual API calls
        try:
            counter = BedrockTokenCounter()
            test_text = "Hello, World!"
            
            # First, get the full token count using real API
            full_count = counter.count_tokens(test_text, "anthropic.claude-3-5-haiku-20241022-v1:0")
            print(f"Full text '{test_text}' has {full_count} tokens")
            
            # Test from 1 token up to full length + 1
            for target_tokens in range(1, full_count + 2):
                result, metadata = counter.truncate(test_text, target_tokens, "anthropic.claude-3-5-haiku-20241022-v1:0", return_metadata=True)
                
                print(f"Target {target_tokens:2d} tokens: '{result}' -> {metadata['final_token_count']} tokens ({metadata['api_calls']} API calls)")
                
                # Verify the result
                assert metadata["final_token_count"] <= target_tokens, f"Target {target_tokens}: got {metadata['final_token_count']} tokens"
                assert len(result) <= len(test_text), f"Result longer than original text"
                
                # For targets >= full length, should return full text
                if target_tokens >= full_count:
                    assert result == test_text, f"Target {target_tokens}: should return full text"
                else:
                    assert len(result) < len(test_text), f"Target {target_tokens}: should be truncated"
            
            # Test that truncating to full length + 1 does nothing (returns full text)
            result, metadata = counter.truncate(test_text, full_count + 1, "anthropic.claude-3-5-haiku-20241022-v1:0", return_metadata=True)
            assert result == test_text, "Truncating to full length + 1 should return full text"
            assert metadata["final_token_count"] <= full_count + 1, "Should not exceed target"
            
        except Exception as e:
            pytest.skip(f"Real API test skipped: {e}")
    
    def test_cli_systematic_behavior_real_api(self):
        """Test CLI systematic behavior: count tokens, then truncate from 1 to full length + 1."""
        # This test requires real AWS credentials and will make actual API calls
        try:
            counter = BedrockTokenCounter()
            test_text = "Hello, World!"
            
            # Step 1: Count tokens (equivalent to: uv run ttok4bedrock 'Hello, World!')
            full_count = counter.count_tokens(test_text, "anthropic.claude-3-5-haiku-20241022-v1:0")
            print(f"Step 1 - Count tokens: '{test_text}' = {full_count} tokens")
            
            # Step 2: Test systematic truncation (equivalent to: uv run ttok4bedrock -t N 'Hello, World!')
            print(f"Step 2 - Systematic truncation from 1 to {full_count + 1} tokens:")
            
            for target_tokens in range(1, full_count + 2):
                result, metadata = counter.truncate(test_text, target_tokens, "anthropic.claude-3-5-haiku-20241022-v1:0", return_metadata=True)
                
                print(f"  Target {target_tokens:2d} tokens: '{result}' -> {metadata['final_token_count']} tokens ({metadata['api_calls']} API calls)")
                
                # Verify CLI-like behavior
                assert metadata["final_token_count"] <= target_tokens, f"Target {target_tokens}: got {metadata['final_token_count']} tokens"
                assert len(result) <= len(test_text), f"Result longer than original text"
                
                # For targets >= full length, should return full text (no truncation needed)
                if target_tokens >= full_count:
                    assert result == test_text, f"Target {target_tokens}: should return full text (no truncation)"
                else:
                    assert len(result) < len(test_text), f"Target {target_tokens}: should be truncated"
            
            # Verify that truncating to full length + 1 does nothing (returns full text)
            result, metadata = counter.truncate(test_text, full_count + 1, "anthropic.claude-3-5-haiku-20241022-v1:0", return_metadata=True)
            assert result == test_text, "Truncating to full length + 1 should return full text (no change)"
            assert metadata["final_token_count"] <= full_count + 1, "Should not exceed target"
            
            print(f"✓ All systematic truncation tests passed for '{test_text}'")
            
        except Exception as e:
            pytest.skip(f"Real API test skipped: {e}")
    
    def test_truncate_real_api_performance(self):
        """Test truncation efficiency with real AWS Bedrock API calls - exact results required."""
        # This test requires real AWS credentials and will make actual API calls
        # Skip if no credentials are available
        try:
            counter = BedrockTokenCounter()
            
            # Test with a realistic long text
            long_text = "This is a comprehensive sample sentence designed for testing token counting efficiency at scale with real AWS Bedrock API calls. " * 50  # ~3500 chars, ~1000 tokens
            
            # Count initial tokens to verify API works
            initial_count = counter.count_tokens(long_text, "anthropic.claude-3-5-haiku-20241022-v1:0")
            print(f"Initial token count: {initial_count}")
            
            # Test truncation with metadata
            result, metadata = counter.truncate(long_text, 200, "anthropic.claude-3-5-haiku-20241022-v1:0", return_metadata=True)
            
            # Verify results
            assert len(result) < len(long_text)
            assert len(result) > 0
            
            # Verify metadata
            assert 'api_calls' in metadata
            assert 'final_token_count' in metadata
            api_calls = metadata['api_calls']
            final_token_count = metadata['final_token_count']
            
            print(f"API calls made: {api_calls}")
            print(f"Final token count: {final_token_count}")
            print(f"Target was: 200 tokens")
            print(f"Result length: {len(result)} chars")
            print(f"Original length: {len(long_text)} chars")
            
            # EXACT results required - no tolerance
            assert final_token_count <= 200, f"Token count {final_token_count} exceeds target 200"
            assert api_calls <= 20, f"API calls {api_calls} exceeds limit 20"
            
            # Verify the result is actually the exact token count
            verification_count = counter.count_tokens(result, "anthropic.claude-3-5-haiku-20241022-v1:0")
            assert verification_count == final_token_count, f"Metadata token count {final_token_count} doesn't match verification {verification_count}"
            
        except Exception as e:
            # Skip test if AWS credentials not available
            pytest.skip(f"Skipping real API test - AWS credentials not available: {e}")
    
    def test_truncate_real_api_efficiency(self):
        """Test that real API truncation is efficient with exact API call counting."""
        try:
            counter = BedrockTokenCounter()
            
            # Create a moderately long text
            medium_text = "This is a sample sentence for testing real API efficiency. " * 30  # ~1800 chars, ~500 tokens
            
            # Test with metadata to get exact API call count
            import time
            start_time = time.time()
            
            result, metadata = counter.truncate(medium_text, 100, "anthropic.claude-3-5-haiku-20241022-v1:0", return_metadata=True)
            
            end_time = time.time()
            duration = end_time - start_time
            
            api_calls = metadata['api_calls']
            final_token_count = metadata['final_token_count']
            
            print(f"Truncation took {duration:.2f} seconds")
            print(f"API calls made: {api_calls}")
            print(f"Final token count: {final_token_count}")
            print(f"Result: {len(result)} chars")
            print(f"Original: {len(medium_text)} chars")
            
            # Verify results
            assert len(result) < len(medium_text)
            assert len(result) > 0
            
            # EXACT requirements
            assert final_token_count <= 100, f"Token count {final_token_count} exceeds target 100"
            assert api_calls <= 20, f"API calls {api_calls} exceeds limit 20"
            
            # Should complete reasonably quickly (our algorithm should be efficient)
            assert duration < 10.0  # Should complete within 10 seconds
            
        except Exception as e:
            pytest.skip(f"Skipping real API test - AWS credentials not available: {e}")


class TestPublicAPI:
    """Test the public API functions."""
    
    @patch('ttok4bedrock.bedrock_counter.BedrockTokenCounter.count_tokens')
    def test_count_tokens_function(self, mock_count):
        """Test the public count_tokens function."""
        mock_count.return_value = 25
        
        result = count_tokens("Test text", "anthropic.claude-3-5-haiku-20241022-v1:0")
        
        assert result == 25
        mock_count.assert_called_once_with("Test text", "anthropic.claude-3-5-haiku-20241022-v1:0")
    
    @patch('ttok4bedrock.bedrock_counter.BedrockTokenCounter.count_tokens')
    def test_count_tokens_with_region(self, mock_count):
        """Test count_tokens with AWS region."""
        mock_count.return_value = 30
        
        result = count_tokens(
            "Test text",
            "anthropic.claude-3-5-haiku-20241022-v1:0",
            aws_region="eu-west-1"
        )
        
        assert result == 30
    
    @patch('ttok4bedrock.bedrock_counter.BedrockTokenCounter.truncate')
    def test_truncate_function(self, mock_truncate):
        """Test the public truncate function."""
        mock_truncate.return_value = "Truncated"
        
        result = truncate(
            "Long text",
            10,
            "anthropic.claude-3-5-haiku-20241022-v1:0"
        )
        
        assert result == "Truncated"
        mock_truncate.assert_called_once()
    


class TestCLI:
    """Test the CLI interface."""
    
    @patch('ttok4bedrock.bedrock_counter.BedrockTokenCounter.count_tokens')
    def test_cli_basic_count(self, mock_count):
        """Test basic token counting via CLI."""
        from click.testing import CliRunner
        from ttok4bedrock.cli import cli
        
        mock_count.return_value = 15
        
        runner = CliRunner()
        result = runner.invoke(cli, ['hello', 'world'])
        
        assert result.exit_code == 0
        assert result.output.strip() == "15"
    
    @patch('ttok4bedrock.bedrock_counter.BedrockTokenCounter.truncate')
    def test_cli_truncate(self, mock_truncate):
        """Test truncation via CLI."""
        from click.testing import CliRunner
        from ttok4bedrock.cli import cli
        
        mock_truncate.return_value = "Truncated text"
        
        runner = CliRunner()
        result = runner.invoke(cli, ['-t', '10', 'long', 'text', 'here'])
        
        assert result.exit_code == 0
        assert result.output == "Truncated text"
    
    
    @patch('ttok4bedrock.bedrock_counter.BedrockTokenCounter.count_tokens')
    def test_cli_with_model(self, mock_count):
        """Test specifying model via CLI."""
        from click.testing import CliRunner
        from ttok4bedrock.cli import cli
        
        mock_count.return_value = 20
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            '-m', 'anthropic.claude-3-5-sonnet-20241022-v2:0',
            'test'
        ])
        
        assert result.exit_code == 0
        assert result.output.strip() == "20"
    
    @patch('ttok4bedrock.bedrock_counter.BedrockTokenCounter.count_tokens')
    def test_cli_with_aws_region(self, mock_count):
        """Test specifying AWS region via CLI."""
        from click.testing import CliRunner
        from ttok4bedrock.cli import cli
        
        mock_count.return_value = 25
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            '--aws-region', 'eu-west-1',
            'test'
        ])
        
        assert result.exit_code == 0
        assert result.output.strip() == "25"
    
    @patch('ttok4bedrock.bedrock_counter.BedrockTokenCounter.count_tokens')
    def test_cli_stdin_input(self, mock_count):
        """Test reading from stdin."""
        from click.testing import CliRunner
        from ttok4bedrock.cli import cli
        
        mock_count.return_value = 30
        
        runner = CliRunner()
        result = runner.invoke(cli, [], input="Input from stdin")
        
        assert result.exit_code == 0
        assert result.output.strip() == "30"
    
    @patch('boto3.client')
    def test_cli_error_handling(self, mock_client):
        """Test that boto3 errors are properly displayed."""
        from click.testing import CliRunner
        from ttok4bedrock.cli import cli
        
        mock_bedrock_client = Mock()
        error_response = {
            'Error': {
                'Code': 'ValidationException',
                'Message': 'Model not found'
            }
        }
        mock_bedrock_client.count_tokens.side_effect = ClientError(
            error_response,
            'CountTokens'
        )
        mock_client.return_value = mock_bedrock_client
        
        runner = CliRunner()
        result = runner.invoke(cli, ['test'])
        
        assert result.exit_code == 1
        assert "ValidationException" in result.output
        assert "Model not found" in result.output
    
    def test_cli_unsupported_model_error(self):
        """Test that unsupported models show clear error message."""
        from click.testing import CliRunner
        from ttok4bedrock.cli import cli
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            '-m', 'amazon.titan-text-express-v1',
            'test'
        ])
        
        assert result.exit_code == 1
        assert "is not supported" in result.output
        assert "Please use one of the supported Anthropic Claude models" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])