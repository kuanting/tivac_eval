# TIVAC: Taiwan Industry Value Chain Benchmark — Evaluation System

TIVAC (Taiwan Industry VAlue Chain) is the first benchmark designed to discover structural knowledge within Taiwan's globally critical industrial ecosystem. TIVAC contains **47 industry chains** and **9,752 expert-verified relationships**, spanning **3,257 unique companies** including 2,363 Taiwan-listed firms and 894 key foreign competitors. The dataset is based on a rigorous hierarchical taxonomy with 151 categories and 534 subcategories.

To evaluate LLMs' domain knowledge, we propose a benchmark with **three fundamental tasks**:
1. **Chain-to-Firms Retrieval** — Given an industry chain, retrieve its member companies
2. **Firm-to-Chains Mapping** — Given a company, identify which value chains it belongs to
3. **Competitor Analysis** — Given a company, identify its competitors across shared chains

We conducted a systematic evaluation of popular commercial and open-source LLMs and revealed that a local RAG system consisting of a lightweight model and well-structured data can outperform commercial models with built-in search functionality. TIVAC is expected to serve as a long-term public resource for researchers in various fields including data mining, knowledge graphs, and applied industrial analytics.

The latest dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/kuantinglai/tivac/).

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Configuration

Copy and edit the environment file:
```bash
cp .env.example .env
# Edit .env with your API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
```

Evaluation settings (prompts, providers, rate limits) are configured in `config/evaluation_config.yaml`.

### 3. Download Latest Dataset

This repository includes a demo version of TIVAC in `datasets/demo/` for quick testing. For the latest full version, download from [Kaggle](https://www.kaggle.com/datasets/kuantinglai/tivac/).

### 4. Run Evaluation

A unified entry point `evaluate.py` is provided at the project root. By default (no subcommand) it runs **direct LLM evaluation**; use the `rag` subcommand for RAG evaluation.

```bash
# Direct LLM evaluation (Firm → Chains) — default mode
python evaluate.py \
    --dataset datasets/demo/qa/firm_chains_qa_local.jsonl \
    --provider openai --model gpt-4o-mini

# Direct LLM evaluation (Chain → Firms)
python evaluate.py \
    --dataset datasets/demo/qa/chain_firms_qa_local.jsonl \
    --provider openai --model gpt-4o-mini

# RAG evaluation
python evaluate.py rag \
    --dataset datasets/demo/qa/firm_chains_qa_local.jsonl \
    --provider openai --model gpt-4o-mini \
    --data-dir datasets/demo/individual_chains

# RAG with Ollama (local, free)
python evaluate.py rag \
    --dataset datasets/demo/qa/firm_chains_qa_local.jsonl \
    --provider ollama --model gemma3:4b \
    --embedding-provider huggingface

# Quick test on first 10 samples
python evaluate.py \
    --dataset datasets/demo/qa/firm_chains_qa_local.jsonl \
    --provider openai --model gpt-4o-mini --max-samples 10

# List available providers and models
python evaluate.py list-models
```

> **Tip:** You can still run the individual scripts directly if needed:
> `python src/evaluation/evaluate_langchain_models.py ...` or
> `python src/evaluation/evaluate_rag_models.py ...`

### 5. Compare Results

```bash
# Quick comparison of two result files
python src/compare_viz/quick_compare.py \
    results/evaluation_results_*.json

# Generate interactive HTML visualization report
python src/compare_viz/visualize_evaluation_results.py \
    --files results/eval1.json results/eval2.json

# Multi-model comparison with charts and rankings
python src/compare_viz/multi_model_visualizer.py \
    results/*.json --output results/comparisons
```

## Three Main Modules

### 1. Evaluation (`src/evaluation/`)
Evaluate LLM models with different retrieval approaches:
- **Baseline** (`evaluate_langchain_models.py`): Direct LLM queries without retrieval
- **RAG** (`evaluate_rag_models.py`): Vector similarity search with FAISS

### 2. Compare & Visualization (`src/compare_viz/`)
Compare and visualize evaluation results:
- **Quick Compare** (`quick_compare.py`): Fast CLI side-by-side metric comparison
- **Visualization** (`visualize_evaluation_results.py`): Interactive HTML reports
- **Multi-Model** (`multi_model_visualizer.py`): N-model comparison with charts and rankings

### 3. Utilities (`src/utils/`)
Shared components:
- **Config** (`config.py`): Dataclasses for model, RAG, and output configurations
- **Metrics** (`metrics.py`): Evaluation metrics (recall, precision, F1, mAP, exact match)
- **Providers** (`providers.py`): LLM provider wrappers (OpenAI, Anthropic, Google, Ollama)

## Dataset Structure

```
datasets/demo/
├── chain_names.json              # Master list of 47 standard chain names
├── individual_chains/            # Raw value chain JSON files (47 chains)
│   ├── 1000_水泥.json
│   ├── D000_半導體.json
│   └── ...
└── qa/                           # QA evaluation datasets
    ├── firm_chains_qa_local.jsonl         # Firm → Chains (local companies)
    ├── firm_chains_qa_foreign.jsonl       # Firm → Chains (foreign companies)
    ├── chain_firms_qa_local.jsonl         # Chain → Firms (local only)
    ├── chain_firms_qa.jsonl              # Chain → Firms (all companies)
    ├── competitors_qa_local.jsonl         # Competitors (local)
    ├── competitors_qa_foreign.jsonl       # Competitors (foreign)
    └── *.meta.json                       # Dataset metadata files
```

## Output

| Module | Output Location | Content |
|--------|-----------------|---------|
| Evaluation | `results/` | JSON evaluation results |
| Compare & Viz | `results/` | HTML reports, PNG charts |
| Multi-Model | `results/comparisons/` | Comparison charts and reports |

## RAG Configuration

The RAG evaluator supports multiple embedding providers:

| Embedding Provider | Model | Cost | Notes |
|-------------------|-------|------|-------|
| `openai` | text-embedding-3-small | Paid | Best quality, requires API key |
| `huggingface` | paraphrase-multilingual-MiniLM-L12-v2 | Free | Good for multilingual, runs locally |
| `ollama` | nomic-embed-text | Free | Requires Ollama running locally |
| `google` | gemini-embedding-001 | Paid | Batch limit ~100 requests |

## Supported LLM Providers

| Provider | Models | API Key Required |
|----------|--------|-----------------|
| **OpenAI** | GPT-4o, GPT-4o-mini, GPT-4, GPT-3.5-turbo | `OPENAI_API_KEY` |
| **Anthropic** | Claude 3.5 Sonnet, Claude 3 Haiku | `ANTHROPIC_API_KEY` |
| **Google** | Gemini 1.5 Pro, Gemini 1.5 Flash | `GOOGLE_API_KEY` |
| **Ollama** | Llama 3.1, Mistral, Qwen2, Gemma3 | None (local) |

## Configuration System

The evaluation system uses a **hybrid configuration approach**:

1. **Dataset Metadata** (`.meta.json` files): Each QA dataset has an accompanying metadata file containing task-specific prompts, chain names, and uncertainty patterns.
2. **Global Config** (`config/evaluation_config.yaml`): Central configuration for providers, rate limits, output settings, and optional prompt overrides.

**Priority**: Dataset metadata is loaded first; global config can override if explicitly set.

| File | Purpose |
|------|---------|
| `config/evaluation_config.yaml` | Global evaluation settings, provider configs, rate limits |
| `datasets/demo/qa/*.meta.json` | Dataset-specific prompts and chain names |
| `datasets/demo/chain_names.json` | Master list of 47 standard chain names |

## Documentation

- [QA Datasets Summary](docs/QA_DATASETS_SUMMARY.md)
- [Chain-Firms QA Details](docs/CHAIN_FIRMS_QA_README.md)