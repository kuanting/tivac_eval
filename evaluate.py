#!/usr/bin/env python3
"""
Unified evaluation entry point for Taiwan Value Chain QA datasets.

Supports two evaluation modes:
  - Direct (default): LLM evaluation without retrieval augmentation
  - RAG:              Retrieval-Augmented Generation evaluation

Usage:
    # Direct evaluation (default — no subcommand needed)
    python evaluate.py --dataset datasets/demo/qa/firm_chains_qa_local.jsonl --provider ollama --model llama3

    # RAG evaluation
    python evaluate.py rag --dataset datasets/demo/qa/firm_chains_qa_local.jsonl --provider ollama --model llama3

    # List available providers and models
    python evaluate.py list-models
"""

import argparse
import asyncio
import sys
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared argument definitions (used by both direct and rag parsers)
# ---------------------------------------------------------------------------

def _add_common_args(parser: argparse.ArgumentParser, required: bool = True) -> None:
    """Add arguments shared between direct and RAG evaluation."""
    parser.add_argument(
        '--dataset', '-d',
        required=required,
        help='Path to JSONL dataset file',
    )
    parser.add_argument(
        '--provider', '-p',
        choices=['openai', 'anthropic', 'google', 'ollama'],
        required=required,
        help='Model provider',
    )
    parser.add_argument(
        '--model', '-m',
        required=required,
        help='Model name',
    )
    parser.add_argument(
        '--config', '-c',
        default=None,
        help='Path to custom YAML config file (default: config/evaluation_config.yaml)',
    )
    parser.add_argument(
        '--api-key', '-k',
        default=None,
        help='API key for the provider',
    )
    parser.add_argument(
        '--base-url',
        default=None,
        help='Base URL for API (useful for Ollama)',
    )
    parser.add_argument(
        '--temperature', '-t',
        type=float,
        default=0.0,
        help='Temperature for generation (default: 0.0)',
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=500,
        help='Maximum tokens for generation (default: 500)',
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=120,
        help='Request timeout in seconds (default: 120)',
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to evaluate (default: all)',
    )
    parser.add_argument(
        '--sample-rate',
        type=float,
        default=1.0,
        help='Rate of samples to evaluate, 0.0–1.0 (default: 1.0)',
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save detailed results to file',
    )
    parser.add_argument(
        '--no-reasoning',
        action='store_true',
        help='Disable reasoning/thinking for models that support it',
    )


def _add_rag_args(parser: argparse.ArgumentParser) -> None:
    """Add RAG-specific arguments."""
    parser.add_argument(
        '--embedding-provider',
        choices=['openai', 'huggingface', 'ollama', 'google'],
        default='openai',
        help='Embedding model provider (default: openai)',
    )
    parser.add_argument(
        '--embedding-model',
        default=None,
        help='Embedding model name (default: provider-specific)',
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of documents to retrieve (default: 5)',
    )
    parser.add_argument(
        '--score-threshold',
        type=float,
        default=0.0,
        help='Minimum similarity score for retrieval (default: 0.0)',
    )
    parser.add_argument(
        '--data-dir',
        default='datasets/demo/individual_chains',
        help='Path to directory containing individual chain JSON files '
             '(default: datasets/demo/individual_chains)',
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_common(args: argparse.Namespace) -> None:
    """Validate shared arguments; exit on failure."""
    if not Path(args.dataset).exists():
        print(f"✗ Error: Dataset file not found: {args.dataset}")
        sys.exit(1)

    if not 0.0 < args.sample_rate <= 1.0:
        print("✗ Error: Sample rate must be between 0.0 and 1.0")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Run direct evaluation
# ---------------------------------------------------------------------------

async def run_direct(args: argparse.Namespace) -> None:
    """Run direct (non-RAG) LLM evaluation."""
    from src.evaluation.evaluate_langchain_models import (
        MultiModelEvaluator,
        ModelConfig,
        EvaluationConfig,
    )

    # Validate required args (not enforced by argparse at top level)
    missing = [name for name in ('dataset', 'provider', 'model') if not getattr(args, name, None)]
    if missing:
        flags = ', '.join(f'--{n}' for n in missing)
        print(f"✗ Error: the following arguments are required: {flags}")
        sys.exit(1)

    _validate_common(args)

    # Load evaluation config
    config_path = Path(args.config) if args.config else None
    eval_config = EvaluationConfig.load(config_path)
    defaults = eval_config.defaults

    # Build model config (use YAML defaults when the user didn't override)
    config = ModelConfig(
        provider=args.provider,
        model_name=args.model,
        temperature=(
            args.temperature
            if args.temperature != 0.0
            else defaults.get('temperature', 0.0)
        ),
        max_tokens=(
            args.max_tokens
            if args.max_tokens != 500
            else defaults.get('max_tokens', 500)
        ),
        timeout=(
            args.timeout
            if args.timeout != 120
            else defaults.get('timeout', 120)
        ),
        api_key=args.api_key,
        base_url=args.base_url,
        enable_reasoning=not args.no_reasoning,
    )

    evaluator = MultiModelEvaluator(config, eval_config)

    results = await evaluator.evaluate_dataset(
        args.dataset,
        max_samples=args.max_samples,
        sample_rate=args.sample_rate,
        save_results=not args.no_save,
    )

    print("\n✓ Direct evaluation completed successfully!")


# ---------------------------------------------------------------------------
# Run RAG evaluation
# ---------------------------------------------------------------------------

async def run_rag(args: argparse.Namespace) -> None:
    """Run RAG evaluation."""
    from src.evaluation.evaluate_rag_models import RAGEvaluator, RAGConfig

    _validate_common(args)

    config = RAGConfig(
        provider=args.provider,
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        api_key=args.api_key,
        base_url=args.base_url,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
        score_threshold=args.score_threshold,
        data_dir=args.data_dir,
        enable_reasoning=not args.no_reasoning,
    )

    evaluator = RAGEvaluator(config)

    results = await evaluator.evaluate_dataset(
        args.dataset,
        max_samples=args.max_samples,
        sample_rate=args.sample_rate,
        save_results=not args.no_save,
    )

    print("\n✓ RAG evaluation completed successfully!")


# ---------------------------------------------------------------------------
# List models
# ---------------------------------------------------------------------------

def run_list_models(args: argparse.Namespace) -> None:
    """Print available providers and models from the evaluation config."""
    from src.evaluation.evaluate_langchain_models import EvaluationConfig

    config_path = Path(args.config) if getattr(args, 'config', None) else None
    eval_config = EvaluationConfig.load(config_path)

    print(f"\n{'=' * 70}")
    print("AVAILABLE PROVIDERS AND MODELS")
    print(f"{'=' * 70}\n")

    for provider, pconfig in eval_config.providers.items():
        print(f"{provider.upper()}:")
        env_key = pconfig.get('env_key')
        if env_key:
            status = "✓" if os.getenv(env_key) else "✗"
            print(f"  Environment: {env_key} {status}")
        else:
            print(f"  Environment: Not required")

        print("  Models:")
        for model in pconfig.get('models', []):
            print(f"    - {model}")
        print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        description='Evaluate LLM models on Taiwan Value Chain QA datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Direct evaluation (default)
  python evaluate.py --dataset datasets/demo/qa/firm_chains_qa_local.jsonl --provider ollama --model llama3

  # RAG evaluation
  python evaluate.py rag --dataset datasets/demo/qa/firm_chains_qa_local.jsonl --provider ollama --model llama3

  # List available models
  python evaluate.py list-models

  # Quick test on 50 samples
  python evaluate.py --dataset datasets/demo/qa/firm_chains_qa_local.jsonl --provider openai --model gpt-4o-mini --max-samples 50
        """,
    )

    # Direct-mode arguments live on the top-level parser so that
    # `python evaluate.py --dataset ... --provider ... --model ...` works
    # without a subcommand.  Not marked required here so that
    # `list-models` can work without them.
    _add_common_args(parser, required=False)

    # Optional subcommands ------------------------------------------------
    subparsers = parser.add_subparsers(dest='mode')

    # rag
    rag_parser = subparsers.add_parser(
        'rag',
        help='Run RAG (Retrieval-Augmented Generation) evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_args(rag_parser)
    _add_rag_args(rag_parser)

    # list-models
    list_parser = subparsers.add_parser(
        'list-models',
        help='List available providers and models from config',
    )
    list_parser.add_argument(
        '--config', '-c',
        default=None,
        help='Path to custom YAML config file',
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == 'list-models':
        run_list_models(args)
    elif args.mode == 'rag':
        asyncio.run(run_rag(args))
    else:
        # Default: direct evaluation
        asyncio.run(run_direct(args))


if __name__ == '__main__':
    main()
