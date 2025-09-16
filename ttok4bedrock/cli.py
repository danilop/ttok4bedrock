"""
CLI interface for ttok4bedrock - ttok compatible.
"""

import click
import sys
from typing import Optional
from .bedrock_counter import BedrockTokenCounter
from botocore.exceptions import ClientError, NoCredentialsError, NoRegionError


@click.command(context_settings={'help_option_names': ['-h', '--help']})
@click.version_option()
@click.argument("prompt", nargs=-1)
@click.option("-i", "--input", "input_file", type=click.File("r"),
              help="Read input from file")
@click.option("-t", "--truncate", type=int,
              help="Truncate to this many tokens")
@click.option("-m", "--model", "model_id", default="anthropic.claude-sonnet-4-20250514-v1:0",
              help="Bedrock model ID")
@click.option("--aws-region", help="AWS region for Bedrock")
def cli(prompt, input_file, truncate, model_id, aws_region):
    """
    Count and truncate text based on tokens using Amazon Bedrock.
    
    Drop-in replacement for ttok with Bedrock model support.
    
    Supports Anthropic Claude models with CountTokens API:
    
    \b
    anthropic.claude-sonnet-4-20250514-v1:0 (default)
    anthropic.claude-3-5-haiku-20241022-v1:0
    anthropic.claude-3-5-sonnet-20241022-v2:0
    anthropic.claude-3-5-sonnet-20240620-v1:0
    anthropic.claude-3-7-sonnet-20250219-v1:0
    anthropic.claude-opus-4-20250514-v1:0
    
    To count tokens for text passed as arguments:
    
        ttok4bedrock one two three
    
    To count tokens from stdin:
    
        cat input.txt | ttok4bedrock
    
    To truncate to 100 tokens:
    
        cat input.txt | ttok4bedrock -t 100
    
    To use a specific Bedrock model:
    
        ttok4bedrock -m anthropic.claude-3-5-sonnet-20241022-v2:0 "text"
    
    To specify AWS region:
    
        ttok4bedrock --aws-region us-west-2 "text"
    """
    
    # Get input text
    if not prompt and input_file is None:
        input_file = sys.stdin
    
    text = " ".join(prompt) if prompt else ""
    
    if input_file is not None:
        input_text = input_file.read()
        if text:
            text = input_text + " " + text
        else:
            text = input_text
    
    if not text:
        click.echo("Error: No input text provided", err=True)
        sys.exit(1)
    
    try:
        # Initialize counter
        counter = BedrockTokenCounter(region=aws_region)
        
        # Handle truncation
        if truncate:
            result_text = counter.truncate(text, truncate, model_id)
            # Output truncated text
            click.echo(result_text, nl=False)
        else:
            # Just count tokens
            token_count = counter.count_tokens(text, model_id)
            # Output just the count
            click.echo(token_count)
    
    except NoCredentialsError:
        click.echo(
            "Error: Unable to locate AWS credentials. "
            "Please configure AWS credentials using aws configure or environment variables.",
            err=True
        )
        sys.exit(1)
    
    except NoRegionError:
        click.echo(
            "Error: No AWS region configured. "
            "Use --aws-region option or configure a default region.",
            err=True
        )
        sys.exit(1)
    
    except ClientError as e:
        # Let boto3 errors bubble up directly
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_message = e.response.get('Error', {}).get('Message', str(e))
        click.echo(f"Error: AWS Bedrock API error ({error_code}): {error_message}", err=True)
        sys.exit(1)
    
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()