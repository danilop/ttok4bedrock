# ttok4bedrock – ttok-style token counting for Amazon Bedrock

Token counting for Amazon Bedrock models - drop-in replacement for **ttok** with exact CLI/SDK compatibility.

## Features

- ✅ **100% ttok-compatible interface** - Drop-in replacement
- 🎯 **Anthropic Claude models** - Uses the Bedrock CountTokens API
- 🔧 **AWS native** - Uses the boto3 default credential/region chain
- 📊 **Accurate counts** - Uses the Bedrock CountTokens API
- ⚡ **Simple and fast** - No caching, no complexity

For more information on this project, see this blog post:

[Token Counting Meets Amazon Bedrock](https://dev.to/aws/token-counting-meets-amazon-bedrock-4dk5)

## Supported Models

**Anthropic Claude models with CountTokens API support.** See the [AWS documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/count-tokens.html) for details.

### Claude Models:
- `anthropic.claude-sonnet-4-20250514-v1:0` (default)
- `anthropic.claude-3-5-haiku-20241022-v1:0`
- `anthropic.claude-3-5-sonnet-20241022-v2:0`
- `anthropic.claude-3-5-sonnet-20240620-v1:0`
- `anthropic.claude-3-7-sonnet-20250219-v1:0`
- `anthropic.claude-opus-4-20250514-v1:0`

## Installation

**Prerequisites:** [Install uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already.

### Option 1: Install from GitHub repository
```bash
uv tool install git+https://github.com/danilop/ttok4bedrock.git
```

### Option 2: Install from local source
```bash
# Clone the repository
git clone https://github.com/danilop/ttok4bedrock.git
cd ttok4bedrock

# Install with uv
uv tool install .
```

### Create a convenient alias
After installation, you can create an alias to use `ttok` instead of `ttok4bedrock`:

```bash
# Add to your shell profile (~/.bashrc, ~/.zshrc, etc.)
alias ttok='ttok4bedrock'

# Or for a one-time use
alias ttok='uv tool run ttok4bedrock'
```

## Quick Start

### CLI Usage (Identical to ttok)

After installation with `uv tool install`, you can use `ttok4bedrock` directly or create an alias for `ttok`:

```bash
# Count tokens (default: Claude Sonnet 4)
ttok4bedrock "Hello, world!"
# Output: 11

# With alias (if you created one)
ttok "Hello, world!"
# Output: 11

# Count from stdin
echo "Count these tokens" | ttok4bedrock
cat document.txt | ttok4bedrock

# Truncate to N tokens
ttok4bedrock -t 100 "Very long text..."
cat large.txt | ttok4bedrock -t 100 > truncated.txt

# Use specific Bedrock model (full model ID)
ttok4bedrock -m anthropic.claude-3-5-sonnet-20241022-v2:0 "Text"
ttok4bedrock -m anthropic.claude-3-7-sonnet-20250219-v1:0 "Text"

# Specify AWS region (uses default if not specified)
ttok4bedrock --aws-region us-west-2 "Text"
```

**Note:** If you haven't installed with `uv tool install`, you can still use `uv run ttok4bedrock` as shown in the migration section below.

## Algorithm Description

### Smart Truncation Algorithm

The truncation algorithm is designed to minimize API calls while achieving perfectly exact token counts. Here's how it works:

#### **Phase 1: Initial Assessment**
1. **Full Text Analysis**: Count tokens for the entire input text
2. **Smart Estimation**: Analyze text characteristics (punctuation density, word length, spacing) to improve initial character-to-token ratio estimation
3. **Target Calculation**: Use the improved ratio to estimate the target character length

#### **Phase 2: Adaptive Learning Loop**
1. **Linear Interpolation**: Start with the estimated character length
2. **Token Measurement**: Count tokens for the estimated text length
3. **Ratio Refinement**: Update the character-to-token ratio based on actual results
4. **Convergence Check**: Continue until the token count is within 1-2 tokens of the target
5. **API Limit Protection**: Self-imposed limit of 20 API calls to prevent runaway loops

#### **Phase 3: Fine-Tuning**
1. **Chunked Adjustment**: Add/remove characters in small chunks (5-10 chars) when close to target
2. **Character-by-Character**: Final precision adjustment (1-5 characters) only when very close to boundary
3. **Exact Boundary**: Find the exact character position where token count crosses the limit

#### **Key Optimizations**
- **Smart Text Analysis**: Accounts for punctuation, word length, and spacing patterns
- **LRU Caching**: Uses `functools.lru_cache` to avoid repeated API calls for identical text
- **Adaptive Learning**: Each API call improves the estimation for subsequent iterations
- **Efficient Convergence**: Typically achieves exact results in 3-5 API calls
- **Overhead Removal**: Automatically subtracts message structure overhead for intuitive token counts

#### **Performance Characteristics**
- **Small texts (≤200 tokens)**: 3-4 API calls, 100% accuracy
- **Medium texts (200-1000 tokens)**: 4-5 API calls, 100% accuracy  
- **Large texts (>1000 tokens)**: 5-17 API calls, 93-100% accuracy
- **Cache hits**: 0.000s (instant) vs 1+ seconds for API calls

### LRU Caching

The library includes intelligent caching to minimize API calls:

- **Automatic Caching**: Uses Python's `functools.lru_cache` for optimal performance
- **Configurable Size**: Default 1000 entries, customizable via constructor
- **Cache Statistics**: Monitor hit rates and performance via `get_cache_info()`
- **Memory Efficient**: Automatic eviction of least recently used entries

```python
# Monitor cache performance
counter = BedrockTokenCounter(cache_size=500)
# ... use counter ...
stats = counter.get_cache_info()
print(f"Cache hit rate: {stats['hit_rate']:.1%}")
```

### Overhead Removal

The library automatically removes message structure overhead to provide intuitive token counts:

- **Message Overhead**: Bedrock API wraps text in message structures that add ~7 tokens
- **Automatic Subtraction**: Token counts show only the actual text content
- **Unified Caching**: Overhead calculation uses the same LRU cache as regular token counting
- **Transparent Operation**: Users see clean token counts without API complexity

```python
# Before: "Hello" = 8 tokens (including message overhead)
# After:  "Hello" = 1 token (text content only)
counter = BedrockTokenCounter()
count = counter.count_tokens("Hello", "anthropic.claude-3-5-haiku-20241022-v1:0")
print(count)  # Output: 1
```

```

### Python SDK Usage (ttok-compatible)

```python
# Import as drop-in replacement
import ttok4bedrock as ttok

# Count tokens
count = ttok.count_tokens("Hello, world!")
print(count)  # 11

# Use specific model (full Bedrock model ID)
count = ttok.count_tokens(
    "Text to count",
    model="anthropic.claude-3-5-sonnet-20241022-v2:0"
)

# Specify AWS region
count = ttok.count_tokens(
    "Text", 
    model="anthropic.claude-3-5-haiku-20241022-v1:0",
    aws_region="eu-west-1"
)

# Truncate text
truncated = ttok.truncate(
    "Very long text...",
    max_tokens=50,
    model="anthropic.claude-3-5-sonnet-20241022-v2:0"
)

```

## AWS Configuration

### Credentials

Uses the standard AWS credential chain (boto3):
1. Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
2. AWS credentials file (`~/.aws/credentials`)
3. AWS profile (`AWS_PROFILE` environment variable)
4. IAM role (for EC2, Lambda, ECS, etc.)

### Region

Order of precedence:
1. `--aws-region` CLI option or `aws_region` parameter
2. `AWS_DEFAULT_REGION` environment variable
3. AWS config file (`~/.aws/config`)
4. Instance metadata (for EC2)

### Required IAM Permissions

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:CountTokens"
      ],
      "Resource": "arn:aws:bedrock:*:*:foundation-model/*"
    }
  ]
}
```

## Migration from ttok

### CLI Migration

```bash
# Before (ttok with OpenAI)
ttok "Count my tokens"
ttok -m gpt-4 "Text"
cat large.txt | ttok -t 100

# After installation with uv tool install
ttok4bedrock "Count my tokens"  # Same interface!
ttok4bedrock -m anthropic.claude-3-5-sonnet-20241022-v2:0 "Text"
cat large.txt | ttok4bedrock -t 100  # Identical usage

# With alias (recommended)
alias ttok='ttok4bedrock'
ttok "Count my tokens"  # Drop-in replacement!
ttok -m anthropic.claude-3-5-sonnet-20241022-v2:0 "Text"
cat large.txt | ttok -t 100

# Alternative: without installation (uv run)
uv run ttok4bedrock "Count my tokens"  # Same interface!
uv run ttok4bedrock -m anthropic.claude-3-5-sonnet-20241022-v2:0 "Text"
cat large.txt | uv run ttok4bedrock -t 100  # Identical usage
```

### Python Migration

```python
# Before (ttok)
import ttok
count = ttok.count_tokens("Text", model="gpt-4")

# After (ttok4bedrock)
import ttok4bedrock as ttok
count = ttok.count_tokens("Text", model="anthropic.claude-3-5-sonnet-20241022-v2:0")
```

## Error Handling

The tool provides clear error messages for AWS issues:

```bash
# Model not found
ttok4bedrock -m anthropic.claude-invalid-model "text"
# Error: AWS Bedrock API error (ValidationException): Model not found

# No credentials
ttok4bedrock "text"
# Error: Unable to locate AWS credentials. Please configure AWS credentials...

# No region
ttok4bedrock "text"
# Error: No AWS region configured. Use --aws-region option or configure a default region.
```

**Note:** If using `uv run ttok4bedrock`, replace `ttok4bedrock` with `uv run ttok4bedrock` in the examples above.

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=ttok4bedrock
```

## License

MIT License (see LICENSE file for details)

## Acknowledgments

- Interface design inspired by [ttok](https://github.com/simonw/ttok) by [Simon Willison](https://twitter.com/simonw)
- Built for the Amazon Bedrock CountTokens API
- See [AWS Bedrock CountTokens documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/count-tokens.html) for supported models and regions
