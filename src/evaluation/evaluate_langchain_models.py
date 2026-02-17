#!/usr/bin/env python3
"""
Multi-Model Evaluation Script for Taiwan Value Chain QA Datasets using LangChain.

This script supports both QA dataset types:
- Firm → Chains: Given a company, predict its industry chains
- Chain → Firms: Given an industry chain, predict its member companies

Supported LLM providers:
- OpenAI (GPT-5.1-chat-latest, GPT-4o, etc.)
- Anthropic (Claude models)
- Google (Gemini models, gemini-3-flash-preview, etc.)
- Ollama (Local models)
- Hugging Face (via transformers)

Features:
1. Auto-detection of dataset type (firm_chains_qa vs chain_firms_qa)
2. Unified interface for different model providers
3. Configurable model parameters
4. Comprehensive evaluation metrics
5. Error handling and retry logic
6. Detailed result analysis and comparison

Usage:
    # Firm → Chains evaluation
    python evaluate_langchain_models.py --dataset datasets/qa/firm_chains_qa_local.jsonl --provider openai --model gpt-4o-mini
    
    # Chain → Firms evaluation  
    python evaluate_langchain_models.py --dataset datasets/qa/chain_firms_qa_large.jsonl --provider openai --model gpt-4o-mini
    
    # Other providers
    python evaluate_langchain_models.py --dataset datasets/qa/firm_chains_qa_local.jsonl --provider anthropic --model claude-3-sonnet-20240229
    python evaluate_langchain_models.py --dataset datasets/qa/chain_firms_qa_large.jsonl --provider google --model gemini-pro
    python evaluate_langchain_models.py --dataset datasets/qa/firm_chains_qa_local.jsonl --provider ollama --model llama2
"""

import json
import argparse
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
import os
import re
from collections import defaultdict
from datetime import datetime
from dotenv import load_dotenv
import warnings
from dataclasses import dataclass, field
import yaml

# Load environment variables from .env file
load_dotenv()

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class EvaluationConfig:
    """Configuration loaded from YAML file."""
    chain_names_file: str = "datasets/chain_names.json"
    prompts: Dict[str, Dict[str, str]] = field(default_factory=dict)
    uncertainty_patterns: List[str] = field(default_factory=list)
    providers: Dict[str, Dict] = field(default_factory=dict)
    defaults: Dict[str, Any] = field(default_factory=dict)
    rate_limiting: Dict[str, float] = field(default_factory=dict)
    output: Dict[str, str] = field(default_factory=dict)
    
    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> 'EvaluationConfig':
        """Load configuration from YAML file."""
        if config_path is None:
            # Default config path relative to workspace root
            workspace_root = Path(__file__).parent.parent.parent
            config_path = workspace_root / "config" / "evaluation_config.yaml"
        
        if not config_path.exists():
            print(f"⚠ Warning: Config file not found: {config_path}")
            print("  Using default configuration.")
            return cls()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            config = cls(
                chain_names_file=data.get('chain_names_file', cls.chain_names_file),
                prompts=data.get('prompts', {}),
                uncertainty_patterns=data.get('uncertainty_patterns', []),
                providers=data.get('providers', {}),
                defaults=data.get('defaults', {}),
                rate_limiting=data.get('rate_limiting', {}),
                output=data.get('output', {})
            )
            print(f"✓ Loaded configuration from {config_path.name}")
            return config
        except Exception as e:
            print(f"⚠ Warning: Failed to load config: {e}")
            return cls()


@dataclass
class DatasetMetadata:
    """Metadata loaded from dataset .meta.json file."""
    dataset_type: str = ""
    dataset_name: str = ""
    description: str = ""
    task: str = ""
    generated_at: str = ""
    source: str = ""
    chain_names: List[str] = field(default_factory=list)
    default_prompt: Dict[str, str] = field(default_factory=dict)
    uncertainty_patterns: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def load(cls, dataset_path: str) -> Optional['DatasetMetadata']:
        """Load metadata from .meta.json file accompanying the dataset."""
        dataset_path = Path(dataset_path)
        meta_path = dataset_path.with_suffix('.meta.json')
        
        if not meta_path.exists():
            print(f"⚠ Warning: Dataset metadata not found: {meta_path}")
            return None
        
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            metadata = cls(
                dataset_type=data.get('dataset_type', ''),
                dataset_name=data.get('dataset_name', ''),
                description=data.get('description', ''),
                task=data.get('task', ''),
                generated_at=data.get('generated_at', ''),
                source=data.get('source', ''),
                chain_names=data.get('chain_names', []),
                default_prompt=data.get('default_prompt', {}),
                uncertainty_patterns=data.get('uncertainty_patterns', []),
                statistics=data.get('statistics', {})
            )
            print(f"✓ Loaded dataset metadata from {meta_path.name}")
            return metadata
        except Exception as e:
            print(f"⚠ Warning: Failed to load dataset metadata: {e}")
            return None

# Import LangChain components
try:
    from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
    from langchain_core.language_models.base import BaseLanguageModel
    from langchain_core.language_models.llms import LLM
    from langchain_core.language_models.chat_models import BaseChatModel
    
    # Provider-specific imports (core providers)
    from langchain_openai import ChatOpenAI, OpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.llms import Ollama
    # from langchain_community.chat_models import ChatOllama
    from langchain_ollama import ChatOllama
    
    # Optional provider imports
    try:
        from langchain_huggingface import HuggingFacePipeline
    except ImportError:
        HuggingFacePipeline = None
    
except ImportError as e:
    print(f"Error: Missing required packages. Install with:")
    print("pip install langchain langchain-openai langchain-anthropic langchain-google-genai langchain-community langchain-huggingface langchain-cohere")
    print(f"Specific error: {e}")
    exit(1)


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    provider: str
    model_name: str
    temperature: float = 0.0
    max_tokens: int = 500
    timeout: int = 120
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    extra_params: Dict[str, Any] = None
    enable_reasoning: bool = True


class ModelProvider:
    """Base class for model providers."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = self._initialize_model()
    
    def _initialize_model(self) -> BaseLanguageModel:
        """Initialize the model based on provider."""
        raise NotImplementedError
    
    async def generate(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Generate response from the model."""
        raise NotImplementedError


class OpenAIProvider(ModelProvider):
    """OpenAI model provider."""
    
    def _initialize_model(self) -> ChatOpenAI:
        api_key = self.config.api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass via --api-key")
        
        extra_params = self.config.extra_params or {}
        
        # Check if this is a "thinking/reasoning" model (o1, o3, GPT-5+, etc.)
        # These models use internal reasoning tokens that count against max_completion_tokens
        model_name = self.config.model_name.lower()
        is_thinking_model = (
            model_name.startswith('o1') or  # o1-preview, o1-mini, o1
            model_name.startswith('o3') or  # o3, o3-mini
            'gpt-5' in model_name or        # GPT-5, GPT-5.1, etc.
            'gpt-6' in model_name or        # Future GPT-6+ models
            'gpt-7' in model_name
        )
        
        # Determine token parameter and value
        max_tokens = self.config.max_tokens
        if is_thinking_model:
            # Thinking models use reasoning tokens (can be 1000-16000+ tokens for reasoning)
            # We need to set max_completion_tokens high enough to accommodate both thinking and output
            # Minimum recommended: 8192 for simple tasks, 16384 for complex tasks
            min_tokens_for_thinking = 8192
            if max_tokens < min_tokens_for_thinking:
                print(f"  ℹ Detected OpenAI thinking model '{self.config.model_name}', "
                      f"increasing max_completion_tokens from {max_tokens} to {min_tokens_for_thinking}")
                max_tokens = min_tokens_for_thinking
            
            # OpenAI thinking models require max_completion_tokens instead of max_tokens
            return ChatOpenAI(
                model=self.config.model_name,
                max_completion_tokens=max_tokens,
                api_key=api_key,
                request_timeout=self.config.timeout,
                **extra_params
            )
        else:
            # Standard models use max_tokens
            return ChatOpenAI(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                api_key=api_key,
                request_timeout=self.config.timeout,
                **extra_params
            )
    
    async def generate(self, messages: List[Dict[str, str]], max_retries: int = 3) -> Optional[str]:
        # Convert to LangChain messages
        lc_messages = []
        for msg in messages:
            if msg["role"] == "system":
                lc_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
        
        last_error = None
        for attempt in range(max_retries):
            try:
                response = await self.model.ainvoke(lc_messages)
                return response.content.strip()
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff: 2, 4, 6 seconds
                    await asyncio.sleep(wait_time)
                    continue
        
        print(f"OpenAI API Error after {max_retries} retries: {last_error}")
        return None


class AnthropicProvider(ModelProvider):
    """Anthropic Claude model provider."""
    
    def _initialize_model(self) -> ChatAnthropic:
        api_key = self.config.api_key or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY environment variable or pass via --api-key")
        
        extra_params = self.config.extra_params or {}
        
        return ChatAnthropic(
            model=self.config.model_name,
            max_tokens=self.config.max_tokens,
            anthropic_api_key=api_key,
            timeout=self.config.timeout,
            **extra_params
        )
    
    async def generate(self, messages: List[Dict[str, str]], max_retries: int = 3) -> Optional[str]:
        # Convert to LangChain messages
        lc_messages = []
        for msg in messages:
            if msg["role"] == "system":
                lc_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
        
        last_error = None
        for attempt in range(max_retries):
            try:
                response = await self.model.ainvoke(lc_messages)
                return response.content.strip()
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    await asyncio.sleep(wait_time)
                    continue
        
        print(f"Anthropic API Error after {max_retries} retries: {last_error}")
        return None


class GoogleProvider(ModelProvider):
    """Google Gemini model provider."""
    
    def _initialize_model(self) -> ChatGoogleGenerativeAI:
        api_key = self.config.api_key or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("Google API key required. Set GOOGLE_API_KEY environment variable or pass via --api-key")
        
        extra_params = self.config.extra_params or {}
        
        # Check if this is a "thinking" model (Pro models, 2.5+)
        # These models use internal reasoning tokens that count against max_output_tokens
        model_name = self.config.model_name.lower()
        is_thinking_model = 'pro' in model_name and ('2.5' in model_name or '3' in model_name)
        
        # Determine max_output_tokens
        max_tokens = self.config.max_tokens
        if is_thinking_model:
            # Pro models use thinking tokens (can be 1000-8000+ tokens for reasoning)
            # We need to set max_output_tokens high enough to accommodate both thinking and output
            # Minimum recommended: 4096 for simple tasks, 8192 for complex tasks
            min_tokens_for_thinking = 4096
            if max_tokens < min_tokens_for_thinking:
                print(f"  ℹ Detected thinking model '{self.config.model_name}', "
                      f"increasing max_output_tokens from {max_tokens} to {min_tokens_for_thinking}")
                max_tokens = min_tokens_for_thinking
        
        return ChatGoogleGenerativeAI(
            model=self.config.model_name,
            max_output_tokens=max_tokens,
            google_api_key=api_key,
            **extra_params
        )
    
    async def generate(self, messages: List[Dict[str, str]], max_retries: int = 3) -> Optional[str]:
        # Convert to LangChain messages
        lc_messages = []
        for msg in messages:
            if msg["role"] == "system":
                lc_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
        
        last_error = None
        for attempt in range(max_retries):
            try:
                response = await self.model.ainvoke(lc_messages)
                return response.content.strip()
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    await asyncio.sleep(wait_time)
                    continue
        
        print(f"Google API Error after {max_retries} retries: {last_error}")
        return None


class OllamaProvider(ModelProvider):
    """Ollama local model provider."""
    
    def _initialize_model(self) -> ChatOllama:
        base_url = self.config.base_url or "http://localhost:11434"
        extra_params = dict(self.config.extra_params or {}) 

        model_lower = (self.config.model_name or "").lower()

        if not self.config.enable_reasoning:
            if "gpt" in model_lower:
                extra_params.setdefault("reasoning", "low")
                return ChatOllama(
                    model=self.config.model_name,
                    base_url=base_url,
                    temperature=self.config.temperature,
                    **extra_params
                )
            else:
                return ChatOllama(
                    model=self.config.model_name,
                    base_url=base_url,
                    reasoning=False,
                    temperature=self.config.temperature,
                    **extra_params
                )
        return ChatOllama(
            model=self.config.model_name,
            base_url=base_url,
            temperature=self.config.temperature,
            **extra_params
        )
    async def generate(self, messages: List[Dict[str, str]], max_retries: int = 3) -> Optional[str]:
        # Convert to LangChain messages
        lc_messages = []
        for msg in messages:
            if msg["role"] == "system":
                lc_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
        
        last_error = None
        for attempt in range(max_retries):
            try:
                response = await self.model.ainvoke(lc_messages)
                return response.content.strip()
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    await asyncio.sleep(wait_time)
                    continue
        
        print(f"Ollama Error after {max_retries} retries: {last_error}")
        return None


class MultiModelEvaluator:
    """Multi-model evaluator for Taiwan value chain QA datasets."""
    
    # Provider class mapping
    PROVIDER_CLASSES = {
        'openai': OpenAIProvider,
        'anthropic': AnthropicProvider,
        'google': GoogleProvider,
        'ollama': OllamaProvider
    }
    
    def __init__(self, config: ModelConfig, eval_config: Optional[EvaluationConfig] = None):
        """Initialize evaluator with model configuration."""
        self.config = config
        self.eval_config = eval_config or EvaluationConfig.load()
        self.provider = self._initialize_provider()
        self.results = []
        self.dataset_type = None  # Will be detected automatically
        self.dataset_metadata = None  # Will be loaded with dataset
        self.chain_names = []  # Will be loaded from metadata or fallback
        self.system_prompt_template = None  # Will be loaded from metadata or config
        self.uncertainty_patterns = []  # Will be loaded from metadata or config
    
    def _load_dataset_metadata(self, dataset_path: str) -> None:
        """Load metadata from dataset and set up chain_names, prompts, etc."""
        # Try to load dataset metadata
        self.dataset_metadata = DatasetMetadata.load(dataset_path)
        
        if self.dataset_metadata:
            # Use chain names from metadata
            self.chain_names = self.dataset_metadata.chain_names
            print(f"✓ Using {len(self.chain_names)} chain names from dataset metadata")
            
            # Use prompt from metadata (can be overridden by config)
            config_prompt = self.eval_config.prompts.get(self.dataset_type, {}).get('system')
            if config_prompt:
                self.system_prompt_template = config_prompt
                print(f"✓ Using system prompt from evaluation config (override)")
            elif self.dataset_metadata.default_prompt.get('system'):
                self.system_prompt_template = self.dataset_metadata.default_prompt['system']
                print(f"✓ Using system prompt from dataset metadata")
            
            # Use uncertainty patterns from metadata (can be overridden by config)
            if self.eval_config.uncertainty_patterns:
                self.uncertainty_patterns = self.eval_config.uncertainty_patterns
                print(f"✓ Using uncertainty patterns from evaluation config (override)")
            elif self.dataset_metadata.uncertainty_patterns:
                self.uncertainty_patterns = self.dataset_metadata.uncertainty_patterns
                print(f"✓ Using uncertainty patterns from dataset metadata")
        else:
            # Fallback to config-only mode
            print("⚠ No dataset metadata found, using fallback configuration")
            self._load_chain_names_fallback()
            
            # Use prompt from config
            config_prompt = self.eval_config.prompts.get(self.dataset_type, {}).get('system')
            if config_prompt:
                self.system_prompt_template = config_prompt
            
            # Use uncertainty patterns from config
            self.uncertainty_patterns = self.eval_config.uncertainty_patterns or [
                '不確定', '無法確定', '不知道', '沒有資料', 
                '無法回答', '資訊不足', '不清楚'
            ]
    
    def _load_chain_names_fallback(self) -> None:
        """Load standard chain names from JSON file (fallback when no metadata)."""
        workspace_root = Path(__file__).parent.parent.parent
        chain_names_path = workspace_root / self.eval_config.chain_names_file
        
        if not chain_names_path.exists():
            print(f"⚠ Warning: Chain names file not found: {chain_names_path}")
            print("  Run 'python scripts/extract_chain_names.py' to generate it.")
            return
        
        try:
            with open(chain_names_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.chain_names = data.get('chain_names', [])
            print(f"✓ Loaded {len(self.chain_names)} chain names from {self.eval_config.chain_names_file} (fallback)")
        except Exception as e:
            print(f"⚠ Warning: Failed to load chain names: {e}")
    
    def _initialize_provider(self) -> ModelProvider:
        """Initialize the appropriate model provider."""
        if self.config.provider not in self.PROVIDER_CLASSES:
            raise ValueError(f"Unsupported provider: {self.config.provider}. Available: {list(self.PROVIDER_CLASSES.keys())}")
        
        provider_class = self.PROVIDER_CLASSES[self.config.provider]
        return provider_class(self.config)
    
    def detect_dataset_type(self, dataset_sample: Dict) -> str:
        """Detect whether this is firm_chains_qa or chain_firms_qa dataset."""
        if 'company' in dataset_sample and 'chains' in dataset_sample:
            return 'competitors_qa'
        # ✅ firm_chains：除了 company 還要有 is_foreign（避免誤判）
        elif 'company' in dataset_sample and 'is_foreign' in dataset_sample:
            return 'firm_chains_qa'
        elif 'chain' in dataset_sample:
            return 'chain_firms_qa'
        else:
            raise ValueError("Unknown dataset format.")
    
    def create_messages(self, question: str, dataset_type: str) -> List[Dict[str, str]]:
        """Create messages for the model based on dataset type."""
        # Use the loaded prompt template (from metadata or config)
        if not self.system_prompt_template:
            raise ValueError(f"No prompt configured for dataset type: {dataset_type}. "
                           "Check dataset metadata or evaluation config.")
        
        # For firm_chains_qa, substitute the chain list
        if dataset_type == 'firm_chains_qa':
            chain_list = "\n".join(f"- {chain}" for chain in self.chain_names)
            system_prompt = self.system_prompt_template.format(chain_list=chain_list)
        else:
            system_prompt = self.system_prompt_template
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    
    async def query_model(self, question: str, entity: str, dataset_type: str, max_retries: int = 3) -> Optional[str]:
        """Query the model with a question."""
        messages = self.create_messages(question, dataset_type)
        
        for attempt in range(max_retries):
            try:
                response = await self.provider.generate(messages)
                if response:
                    return response
                
            except Exception as e:
                print(f"  ⚠ Error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return None
    
    def parse_response(self, response: str, dataset_type: str) -> List[str]:
        """Parse the model's response to extract chain names or company names."""
        if not response:
            return []

        # Check for uncertainty responses
        uncertainty_patterns = [
            r'不確定', r'無法確定', r'不知道', r'沒有資料', r'無法回答',
            r'資訊不足', r'不清楚', r'根據提供的資料無法確定'
        ]
        for pattern in uncertainty_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return []

        # ----------------------------
        # NEW: Try parse JSON outputs (competitors/chain_firms often return JSON)
        # ----------------------------
        if dataset_type in ("chain_firms_qa", "competitors_qa"):
            raw = response.strip()

            # remove code fences if any
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"\s*```$", "", raw)

            # Some models may add leading/trailing text; try to extract JSON object/array
            # Find first "{" or "[" and last matching "}" or "]"
            start_idx = None
            for ch in ["{", "["]:
                i = raw.find(ch)
                if i != -1:
                    start_idx = i if start_idx is None else min(start_idx, i)
            if start_idx is not None:
                # heuristic end
                end_idx = max(raw.rfind("}"), raw.rfind("]"))
                if end_idx != -1 and end_idx > start_idx:
                    candidate = raw[start_idx:end_idx+1]
                else:
                    candidate = raw
            else:
                candidate = raw

            try:
                import json as _json
                obj = _json.loads(candidate)

                extracted: List[str] = []

                # If it's a dict like {"companies":[...]} / {"competitors":[...]}
                if isinstance(obj, dict):
                    # common keys you might see
                    for key in ["companies", "company", "competitors", "answer", "predicted", "predicted_answer"]:
                        if key in obj and isinstance(obj[key], list):
                            extracted = obj[key]
                            break

                    # fallback: first list value in dict
                    if not extracted:
                        for v in obj.values():
                            if isinstance(v, list):
                                extracted = v
                                break

                # If it's a list directly: ["A","B",...]
                elif isinstance(obj, list):
                    extracted = obj

                if extracted:
                    cleaned = []
                    for x in extracted:
                        if not isinstance(x, str):
                            continue
                        x = x.strip()
                        # strip surrounding quotes / punctuation
                        x = x.strip(' \t\r\n"\'')
                        x = re.sub(r'[，。、；：]$', '', x)
                        if len(x) >= 2:
                            cleaned.append(x)
                    return cleaned

            except Exception:
                # If JSON parsing fails, fall back to line parsing below
                pass

        # ----------------------------
        # Existing line-based parsing
        # ----------------------------
        lines = response.split('\n')
        items = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Remove common prefixes (numbers, bullets, dashes)
            line = re.sub(r'^[\d\.\-\*\•\→]+\s*', '', line)
            line = re.sub(r'^\s*[-·]\s*', '', line)

            # Remove trailing punctuation
            line = re.sub(r'[，。、；：]$', '', line)

            if len(line) < 2:
                continue

            if dataset_type == 'firm_chains_qa':
                if '產業鏈' in line:
                    items.append(line.strip())
            elif dataset_type in ('chain_firms_qa', 'competitors_qa'):
                if re.search(r'[\u4e00-\u9fff]', line) or re.search(r'[A-Za-z]', line):
                    items.append(line.strip())

        return items

    
    def calculate_metrics(self, predicted: List[str], actual: List[str]) -> Dict[str, float]:
        """Calculate evaluation metrics for a single prediction."""
        # Convert to sets for comparison (case-insensitive)
        pred_set = set(p.strip() for p in predicted)
        actual_set = set(a.strip() for a in actual)
        
        # Calculate intersection
        correct = pred_set & actual_set
        correct_count = len(correct)
        
        # Calculate metrics
        recall = correct_count / len(actual_set) if actual_set else 0.0
        precision = correct_count / len(pred_set) if pred_set else 0.0
        
        # F1 score
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        # Exact match
        exact_match = 1.0 if pred_set == actual_set else 0.0
        
        return {
            'recall': recall,
            'precision': precision,
            'f1': f1,
            'exact_match': exact_match,
            'correct_count': correct_count,
            'predicted_count': len(pred_set),
            'actual_count': len(actual_set),
            'false_positives': len(pred_set - actual_set),
            'false_negatives': len(actual_set - pred_set)
        }
    
    def calculate_average_precision(self, predicted: List[str], actual: List[str]) -> float:
        """Calculate Average Precision for a single query."""
        if not actual or not predicted:
            return 0.0
        
        actual_set = set(actual)
        num_correct = 0
        sum_precision = 0.0
        
        for i, pred in enumerate(predicted, 1):
            if pred in actual_set:
                num_correct += 1
                precision_at_i = num_correct / i
                sum_precision += precision_at_i
        
        return sum_precision / len(actual_set) if actual_set else 0.0
    
    async def evaluate_dataset(self, dataset_path: str, max_samples: Optional[int] = None,
                              sample_rate: float = 1.0, save_results: bool = True) -> Dict:
        """Evaluate the model on the entire dataset."""
        
        # Load dataset and detect type
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f]
        
        # Auto-detect dataset type from first sample
        self.dataset_type = self.detect_dataset_type(dataset[0])
        
        # Load dataset metadata (chain_names, prompts, etc.)
        self._load_dataset_metadata(dataset_path)
        
        print(f"\n{'='*70}")
        if self.dataset_type == 'firm_chains_qa':
            print(f"Evaluating {self.config.provider.upper()} {self.config.model_name} on Firm→Chains Dataset")
        elif self.dataset_type == 'chain_firms_qa':
            print(f"Evaluating {self.config.provider.upper()} {self.config.model_name} on Chain→Firms Dataset")
        else:   
            print(f"Evaluating {self.config.provider.upper()} {self.config.model_name} on Competitors Dataset")
        print(f"{'='*70}\n")
        print(f"Dataset: {dataset_path}")
        print(f"Dataset Type: {self.dataset_type}")
        print(f"Provider: {self.config.provider}")
        print(f"Model: {self.config.model_name}")
        print(f"Temperature: {self.config.temperature}")
        
        total_samples = len(dataset)
        
        # Apply sampling
        if sample_rate < 1.0:
            import random
            random.seed(42)
            dataset = random.sample(dataset, int(total_samples * sample_rate))
        
        if max_samples:
            dataset = dataset[:max_samples]
        
        print(f"Total samples: {total_samples}")
        print(f"Evaluating: {len(dataset)} samples")
        print(f"\nStarting evaluation...\n")
        
        # Evaluation metrics
        results = []
        total_recall = 0.0
        total_precision = 0.0
        total_f1 = 0.0
        total_ap = 0.0
        total_exact_match = 0
        
        # Category-specific metrics
        by_answer_count = defaultdict(lambda: {
            'count': 0, 'recall': 0.0, 'precision': 0.0, 'f1': 0.0
        })
        
        # For firm_chains_qa: by foreign status; for chain_firms_qa: by local/foreign company counts
        by_category = {}
        if self.dataset_type == 'firm_chains_qa':
            by_category = {
                True: {'count': 0, 'recall': 0.0, 'precision': 0.0, 'f1': 0.0},   # is_foreign=True
                False: {'count': 0, 'recall': 0.0, 'precision': 0.0, 'f1': 0.0}   # is_foreign=False
            }
        else:  # chain_firms_qa and competitors_qa
            by_category = {
                'high_local': {'count': 0, 'recall': 0.0, 'precision': 0.0, 'f1': 0.0},    # >70% local companies
                'mixed': {'count': 0, 'recall': 0.0, 'precision': 0.0, 'f1': 0.0},         # 30-70% local companies
                'high_foreign': {'count': 0, 'recall': 0.0, 'precision': 0.0, 'f1': 0.0}   # <30% local companies
            }
        
        # Track errors
        error_analysis = {
            'api_errors': 0,
            'empty_responses': 0,
            'parse_errors': 0
        }
        
        start_time = time.time()
        
        for idx, item in enumerate(dataset, 1):
            question = item['question']
            actual_answer = item['answer']
            answer_count = item['answer_count']
            
            if self.dataset_type == 'firm_chains_qa':
                entity = item['company']
                is_foreign = item['is_foreign']
                category_key = is_foreign
                print(f"[{idx}/{len(dataset)}] {entity} ({answer_count} chains)...", end=' ')
            elif self.dataset_type == 'chain_firms_qa': 
                entity = item['chain']
                local_count = item.get('local_count', 0)
                foreign_count = item.get('foreign_count', 0)
                total_companies = local_count + foreign_count
                local_ratio = local_count / total_companies if total_companies > 0 else 0
                
                if local_ratio > 0.7:
                    category_key = 'high_local'
                elif local_ratio < 0.3:
                    category_key = 'high_foreign'
                else:
                    category_key = 'mixed'
                    
                print(f"[{idx}/{len(dataset)}] {entity} ({answer_count} companies, {local_count}L/{foreign_count}F)...", end=' ')
            else:  #  competitors_qa
                entity = item.get('company', 'UNKNOWN')
                local_count = item.get('local_count', 0)
                foreign_count = item.get('foreign_count', 0)
                total = local_count + foreign_count
                local_ratio = local_count / total if total > 0 else 0

                if local_ratio > 0.7:
                    category_key = 'high_local'
                elif local_ratio < 0.3:
                    category_key = 'high_foreign'
                else:
                    category_key = 'mixed'

                print(f"[{idx}/{len(dataset)}] {entity} ({item['answer_count']} competitors, {local_count}L/{foreign_count}F)...", end=' ')

            # Query model
            response = await self.query_model(question, entity, self.dataset_type)
            
            if response is None:
                print("❌ API Error")
                error_analysis['api_errors'] += 1
                predicted_answer = []
            else:
                # Parse response
                predicted_answer = self.parse_response(response, self.dataset_type)
                
                if not predicted_answer:
                    error_analysis['empty_responses'] += 1
                
                print(f"✓ ({len(predicted_answer)} predicted)")
            
            # Calculate metrics
            metrics = self.calculate_metrics(predicted_answer, actual_answer)
            ap = self.calculate_average_precision(predicted_answer, actual_answer)
            
            # Store result
            result = {
                'index': idx,
                'entity': entity,
                'question': question,
                'actual_answer': actual_answer,
                'predicted_answer': predicted_answer,
                'response': response,
                'metrics': metrics,
                'average_precision': ap,
                'dataset_type': self.dataset_type
            }
            
            # Add dataset-specific fields
            if self.dataset_type == 'firm_chains_qa':
                result['company'] = entity
                result['is_foreign'] = is_foreign
                result['actual_chains'] = actual_answer
                result['predicted_chains'] = predicted_answer
            elif self.dataset_type == 'chain_firms_qa': 
                result['chain'] = entity
                result['local_count'] = local_count
                result['foreign_count'] = foreign_count
                result['actual_companies'] = actual_answer
                result['predicted_companies'] = predicted_answer
            else:  # competitors_qa
                result['company'] = entity
                result['local_count'] = local_count
                result['foreign_count'] = foreign_count
                result['actual_competitors'] = actual_answer
                result['predicted_competitors'] = predicted_answer
                        
            results.append(result)
            
            # Aggregate metrics
            total_recall += metrics['recall']
            total_precision += metrics['precision']
            total_f1 += metrics['f1']
            total_ap += ap
            if metrics['exact_match'] == 1.0:
                total_exact_match += 1
            
            # Category-specific metrics
            by_answer_count[answer_count]['count'] += 1
            by_answer_count[answer_count]['recall'] += metrics['recall']
            by_answer_count[answer_count]['precision'] += metrics['precision']
            by_answer_count[answer_count]['f1'] += metrics['f1']
            
            by_category[category_key]['count'] += 1
            by_category[category_key]['recall'] += metrics['recall']
            by_category[category_key]['precision'] += metrics['precision']
            by_category[category_key]['f1'] += metrics['f1']
            
            # Rate limiting for API providers
            if self.config.provider != 'ollama':
                api_delay = self.eval_config.rate_limiting.get('api_delay', 0.1)
                await asyncio.sleep(api_delay)
        
        elapsed_time = time.time() - start_time
        
        # Calculate average metrics
        n = len(dataset)
        avg_metrics = {
            'recall': total_recall / n,
            'precision': total_precision / n,
            'f1': total_f1 / n,
            'mAP': total_ap / n,
            'exact_match_rate': total_exact_match / n,
            'evaluated_samples': n,
            'total_samples': total_samples,
            'elapsed_time': elapsed_time,
            'avg_time_per_sample': elapsed_time / n
        }
        
        # Category-specific averages
        for count_key in by_answer_count:
            cat = by_answer_count[count_key]
            if cat['count'] > 0:
                cat['recall'] /= cat['count']
                cat['precision'] /= cat['count']
                cat['f1'] /= cat['count']
        
        for category_key in by_category:
            cat = by_category[category_key]
            if cat['count'] > 0:
                cat['recall'] /= cat['count']
                cat['precision'] /= cat['count']
                cat['f1'] /= cat['count']
        
        # Compile full results
        full_results = {
            'provider': self.config.provider,
            'model': self.config.model_name,
            'temperature': self.config.temperature,
            'dataset': str(dataset_path),
            'dataset_type': self.dataset_type,
            'timestamp': datetime.now().isoformat(),
            'average_metrics': avg_metrics,
            'by_answer_count': dict(by_answer_count),
            'by_category': by_category,
            'error_analysis': error_analysis,
            'detailed_results': results
        }
        
        # Save results
        if save_results:
            self._save_results(full_results, dataset_path)
        
        # Print summary
        self._print_summary(full_results)
        
        return full_results
    
    def _save_results(self, results: Dict, dataset_path: str):
        """Save evaluation results to JSON file."""
        dataset_name = Path(dataset_path).stem
        
        # Get output settings from config
        output_config = self.eval_config.output
        timestamp_format = output_config.get('timestamp_format', "%Y%m%d_%H%M%S")
        results_dir = output_config.get('results_dir', 'results')
        
        timestamp = datetime.now().strftime(timestamp_format)
        model_name = self.config.model_name.replace("/", "_").replace(":", "_")
        
        # Save to results directory from config
        workspace_root = Path(__file__).parent.parent.parent
        output_dir = workspace_root / results_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"evaluation_results_{dataset_name}_{self.config.provider}_{model_name}_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Detailed results saved to: {output_file}")
    
    def _print_summary(self, results: Dict):
        """Print evaluation summary."""
        metrics = results['average_metrics']
        by_count = results['by_answer_count']
        by_category = results['by_category']
        errors = results['error_analysis']
        dataset_type = results['dataset_type']
        
        print(f"\n{'='*70}")
        print("EVALUATION SUMMARY")
        print(f"{'='*70}\n")
        
        print(f"Provider: {results['provider']}")
        print(f"Model: {results['model']}")
        print(f"Dataset Type: {dataset_type}")
        print(f"Evaluated: {metrics['evaluated_samples']} samples")
        print(f"Time: {metrics['elapsed_time']:.1f}s ({metrics['avg_time_per_sample']:.2f}s per sample)")
        
        print(f"\n{'='*70}")
        print("OVERALL METRICS")
        print(f"{'='*70}")
        print(f"  Recall:            {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"  Precision:         {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"  F1 Score:          {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
        print(f"  mAP:               {metrics['mAP']:.4f} ({metrics['mAP']*100:.2f}%)")
        print(f"  Exact Match Rate:  {metrics['exact_match_rate']:.4f} ({metrics['exact_match_rate']*100:.2f}%)")
        
        print(f"\n{'='*70}")
        if dataset_type == 'firm_chains_qa':
            print("BY COMPANY TYPE")
        elif dataset_type == 'chain_firms_qa':
            print("BY CHAIN COMPOSITION")
        else:  # competitors_qa
            print("BY COMPETITOR SET COMPOSITION")
        print(f"{'='*70}")
        
        if dataset_type == 'firm_chains_qa':
            print(f"  Local Companies (n={by_category[False]['count']}):")
            print(f"    Recall:    {by_category[False]['recall']:.4f}")
            print(f"    Precision: {by_category[False]['precision']:.4f}")
            print(f"    F1 Score:  {by_category[False]['f1']:.4f}")
            print(f"\n  Foreign Companies (n={by_category[True]['count']}):")
            print(f"    Recall:    {by_category[True]['recall']:.4f}")
            print(f"    Precision: {by_category[True]['precision']:.4f}")
            print(f"    F1 Score:  {by_category[True]['f1']:.4f}")
        else: # chain_firms_qa and competitors_qa    
            print(f"  High Local (>70% local, n={by_category['high_local']['count']}):")
            print(f"    Recall:    {by_category['high_local']['recall']:.4f}")
            print(f"    Precision: {by_category['high_local']['precision']:.4f}")
            print(f"    F1 Score:  {by_category['high_local']['f1']:.4f}")
            print(f"\n  Mixed (30-70% local, n={by_category['mixed']['count']}):")
            print(f"    Recall:    {by_category['mixed']['recall']:.4f}")
            print(f"    Precision: {by_category['mixed']['precision']:.4f}")
            print(f"    F1 Score:  {by_category['mixed']['f1']:.4f}")
            print(f"\n  High Foreign (<30% local, n={by_category['high_foreign']['count']}):")
            print(f"    Recall:    {by_category['high_foreign']['recall']:.4f}")
            print(f"    Precision: {by_category['high_foreign']['precision']:.4f}")
            print(f"    F1 Score:  {by_category['high_foreign']['f1']:.4f}")
        
        print(f"\n{'='*70}")
        if dataset_type == 'firm_chains_qa':
            print("BY CHAIN COUNT")
        elif dataset_type == 'chain_firms_qa':
            print("BY COMPANY COUNT")
        else:
            print("BY COMPETITOR SET COMPOSITION")
        print(f"{'='*70}")
        for count in sorted(by_count.keys())[:10]:  # Show first 10
            cat = by_count[count]
            if dataset_type == 'firm_chains_qa':
                item_type = "chain(s)"
            elif dataset_type == 'chain_firms_qa':
                item_type = "companies"
            else:
                item_type = "competitors"
            print(f"  {count} {item_type} (n={cat['count']}):")
            print(f"    Recall: {cat['recall']:.4f}, Precision: {cat['precision']:.4f}, F1: {cat['f1']:.4f}")
        
        print(f"\n{'='*70}")
        print("ERROR ANALYSIS")
        print(f"{'='*70}")
        print(f"  API Errors:       {errors['api_errors']}")
        print(f"  Empty Responses:  {errors['empty_responses']}")
        print(f"  Parse Errors:     {errors['parse_errors']}")
        
        # Show some example errors
        print(f"\n{'='*70}")
        print("EXAMPLE PREDICTIONS")
        print(f"{'='*70}")
        
        detailed = results['detailed_results']
        
        # Show perfect predictions
        perfect = [r for r in detailed if r['metrics']['exact_match'] == 1.0]
        if perfect:
            print(f"\n✓ Perfect Predictions (showing first 3 of {len(perfect)}):")
            for r in perfect[:3]:
                entity_name = r.get('company', r.get('chain', r['entity']))
                predicted = r.get('predicted_chains',
                    r.get('predicted_companies',
                    r.get('predicted_competitors', r['predicted_answer'])))
                print(f"\n  Entity: {entity_name}")
                print(f"  Predicted: {predicted}")
        
        # Show partial matches
        partial = [r for r in detailed if 0 < r['metrics']['recall'] < 1.0]
        if partial:
            print(f"\n⚠ Partial Matches (showing first 3 of {len(partial)}):")
            for r in partial[:3]:
                entity_name = r.get('company', r.get('chain', r['entity']))
                actual = r.get('actual_chains', r.get('actual_companies', r.get('actual_competitors', r['actual_answer'])))
                predicted = r.get('predicted_chains', r.get('predicted_companies', r.get('predicted_competitors', r['predicted_answer'])))
                print(f"\n  Entity: {entity_name}")
                print(f"  Actual:    {actual}")
                print(f"  Predicted: {predicted}")
                print(f"  Recall: {r['metrics']['recall']:.2f}, Precision: {r['metrics']['precision']:.2f}")
        
        # Show complete misses
        misses = [r for r in detailed if r['metrics']['recall'] == 0.0]
        if misses:
            print(f"\n✗ Complete Misses (showing first 3 of {len(misses)}):")
            for r in misses[:3]:
                entity_name = r.get('company', r.get('chain', r['entity']))
                actual = r.get('actual_chains', r.get('actual_companies', r.get('actual_competitors', r['actual_answer'])))
                predicted = r.get('predicted_chains', r.get('predicted_companies', r.get('predicted_competitors', r['predicted_answer'])))
                print(f"\n  Entity: {entity_name}")
                print(f"  Actual:    {actual}")
                print(f"  Predicted: {predicted if predicted else '(empty)'}")
        
        print(f"\n{'='*70}\n")


def list_available_models():
    """List all available providers and models."""
    print(f"\n{'='*70}")
    print("AVAILABLE PROVIDERS AND MODELS")
    print(f"{'='*70}\n")
    
    # Load config from YAML
    eval_config = EvaluationConfig.load()
    
    for provider, config in eval_config.providers.items():
        print(f"{provider.upper()}:")
        env_key = config.get('env_key')
        if env_key:
            env_status = "✓" if os.getenv(env_key) else "✗"
            print(f"  Environment: {env_key} {env_status}")
        else:
            print(f"  Environment: Not required")
        
        print(f"  Models:")
        for model in config.get('models', []):
            print(f"    - {model}")
        print()


async def main():
    # First parse just to check --config and --list-models early
    import sys
    config_path = None
    
    # Quick check for --config argument before full parsing
    for i, arg in enumerate(sys.argv):
        if arg in ('--config', '-c') and i + 1 < len(sys.argv):
            config_path = Path(sys.argv[i + 1])
            break
    
    # Load evaluation config from YAML (custom or default)
    eval_config = EvaluationConfig.load(config_path)
    
    # Get available providers from config
    available_providers = list(eval_config.providers.keys())
    
    parser = argparse.ArgumentParser(
        description='Multi-Model Evaluation for Taiwan Value Chain QA datasets using LangChain',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Firm→Chains evaluation (OpenAI GPT-4o-mini)
  python evaluate_langchain_models.py --dataset datasets/qa/firm_chains_qa_local.jsonl --provider openai --model gpt-4o-mini
  
  # Chain→Firms evaluation (OpenAI GPT-4o-mini)  
  python evaluate_langchain_models.py --dataset datasets/qa/chain_firms_qa_large.jsonl --provider openai --model gpt-4o-mini
  
  # Anthropic Claude
  python evaluate_langchain_models.py --dataset datasets/qa/firm_chains_qa_local.jsonl --provider anthropic --model claude-3-sonnet-20240229
  
  # Google Gemini
  python evaluate_langchain_models.py --dataset datasets/qa/chain_firms_qa_large.jsonl --provider google --model gemini-pro
  
  # Local Ollama model
  python evaluate_langchain_models.py --dataset datasets/qa/firm_chains_qa_local.jsonl --provider ollama --model llama2
  
  # Quick test on 50 samples
  python evaluate_langchain_models.py --dataset datasets/qa/chain_firms_qa_large.jsonl --provider openai --model gpt-4o-mini --max-samples 50
  
  # Use custom config file
  python evaluate_langchain_models.py --dataset datasets/qa/firm_chains_qa_local.jsonl --provider openai --model gpt-4o-mini --config custom_config.yaml
  
  # List available models
  python evaluate_langchain_models.py --list-models
        """
    )
    
    parser.add_argument('--dataset', '-d',
                       help='Path to JSONL dataset file')
    parser.add_argument('--provider', '-p', choices=available_providers,
                       help='Model provider')
    parser.add_argument('--model', '-m',
                       help='Model name')
    parser.add_argument('--config', '-c',
                       help='Path to custom YAML config file (default: config/evaluation_config.yaml)')
    parser.add_argument('--api-key', '-k',
                       help='API key for the provider')
    parser.add_argument('--base-url',
                       help='Base URL for API (useful for Ollama)')
    parser.add_argument('--temperature', '-t', type=float, default=0.0,
                       help='Temperature for generation (default: from config)')
    parser.add_argument('--max-tokens', type=int, default=500,
                       help='Maximum tokens for generation (default: from config)')
    parser.add_argument('--timeout', type=int, default=120,
                       help='Request timeout in seconds (default: from config)')
    parser.add_argument('--max-samples', type=int,
                       help='Maximum number of samples to evaluate (default: all)')
    parser.add_argument('--sample-rate', type=float, default=1.0,
                       help='Rate of samples to evaluate, 0.0-1.0 (default: 1.0)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save detailed results to file')
    parser.add_argument('--list-models', action='store_true',
                       help='List available providers and models')
    # resoning switch
    parser.add_argument(
        '--no-reasoning',
        action='store_true',
        help='Disable reasoning for Ollama models (default: reasoning enabled)'
    )
    
    args = parser.parse_args()
    
    # List models and exit
    if args.list_models:
        list_available_models()
        return
    
    # Validate required arguments
    if not args.dataset or not args.provider or not args.model:
        print("✗ Error: --dataset, --provider, and --model are required (unless using --list-models)")
        parser.print_help()
        exit(1)
    
    # Validate dataset path
    if not Path(args.dataset).exists():
        print(f"✗ Error: Dataset file not found: {args.dataset}")
        exit(1)
    
    # Validate sample rate
    if not 0.0 < args.sample_rate <= 1.0:
        print("✗ Error: Sample rate must be between 0.0 and 1.0")
        exit(1)
    
    # Create model configuration using defaults from YAML
    defaults = eval_config.defaults
    config = ModelConfig(
        provider=args.provider,
        model_name=args.model,
        temperature=args.temperature if args.temperature != 0.0 else defaults.get('temperature', 0.0),
        max_tokens=args.max_tokens if args.max_tokens != 500 else defaults.get('max_tokens', 500),
        timeout=args.timeout if args.timeout != 120 else defaults.get('timeout', 120),
        api_key=args.api_key,
        base_url=args.base_url,
        enable_reasoning=(not args.no_reasoning)
    )
    
    # Create evaluator with eval_config
    try:
        evaluator = MultiModelEvaluator(config, eval_config)
    except ValueError as e:
        print(f"✗ Error: {e}")
        exit(1)
    
    # Run evaluation
    try:
        results = await evaluator.evaluate_dataset(
            args.dataset,
            max_samples=args.max_samples,
            sample_rate=args.sample_rate,
            save_results=not args.no_save
        )
        
        print("✓ Evaluation completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Evaluation interrupted by user.")
        exit(1)
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == '__main__':
    asyncio.run(main())