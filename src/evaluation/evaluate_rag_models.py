#!/usr/bin/env python3
"""
RAG Evaluation Script for Taiwan Value Chain QA Datasets using LangChain.

This script evaluates RAG (Retrieval-Augmented Generation) performance on Taiwan value chain
datasets by using the raw JSONL data as the knowledge base for retrieval.

RAG Components:
1. Knowledge Base: Taiwan value chain data loaded from individual chain JSON files
2. Retrieval: Vector similarity search using embeddings
3. Generation: LLM with retrieved context

Supported LLM providers:
- OpenAI (GPT-3.5, GPT-4, GPT-4o, etc.)
- Anthropic (Claude models)
- Google (Gemini models)
- Ollama (Local models)

Features:
1. Vector store creation from Taiwan value chain data
2. Similarity-based retrieval for relevant context
3. RAG-enhanced answer generation
4. Comprehensive evaluation metrics
5. Comparison with baseline (no-RAG) performance

Usage:
    # RAG evaluation with OpenAI
    python evaluate_rag_models.py --dataset datasets/qa/firm_chains_qa_local.jsonl --provider openai --model gpt-4o-mini
    
    # RAG evaluation with Ollama
    python evaluate_rag_models.py --dataset datasets/qa/firm_chains_qa_local.jsonl --provider ollama --model gemma3:4b
    
    # Custom retrieval parameters
    python evaluate_rag_models.py --dataset datasets/qa/firm_chains_qa_local.jsonl --provider openai --model gpt-4o-mini --top-k 5 --score-threshold 0.7
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
from dataclasses import dataclass
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Import LangChain components
try:
    from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
    from langchain_core.language_models.base import BaseLanguageModel
    from langchain_core.language_models.llms import LLM
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.documents import Document
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    
    # Vector store and embeddings
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    # from langchain_community.embeddings import OllamaEmbeddings
    from langchain_ollama import OllamaEmbeddings
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    
    # Provider-specific imports
    from langchain_openai import ChatOpenAI, OpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.llms import Ollama
    # from langchain_community.chat_models import ChatOllama
    from langchain_ollama import ChatOllama
    
except ImportError as e:
    print(f"Error: Missing required packages. Install with:")
    print("pip install langchain langchain-openai langchain-anthropic langchain-google-genai langchain-community langchain-huggingface faiss-cpu")
    print(f"Specific error: {e}")
    exit(1)


@dataclass
class RAGConfig:
    """Configuration for RAG system."""
    provider: str
    model_name: str
    temperature: float = 0.0
    max_tokens: int = 500
    timeout: int = 30
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    # reasoning switch
    enable_reasoning: bool = True
    # RAG-specific parameters
    embedding_provider: str = "openai"  # "openai", "huggingface", "ollama", "google"
    embedding_model: str = None  # Will use provider-specific default if None
    top_k: int = 5
    score_threshold: float = 0.0
    chunk_size: int = 1000
    chunk_overlap: int = 200
    data_dir: str = "datasets/demo/individual_chains"
    
    extra_params: Dict[str, Any] = None


class TaiwanValueChainKnowledgeBase:
    """Knowledge base for Taiwan value chain data."""
    
    def __init__(self, data_dir: str = "datasets/demo/individual_chains"):
        """Initialize knowledge base from Taiwan value chain data."""
        self.data_dir = Path(data_dir)
        self.documents = []
        self.company_to_chains = {}
        self.chain_to_companies = {}
        self._load_data()
    
    def _load_data(self):
        """Load Taiwan value chain data from JSON files."""
        print(f"Loading Taiwan value chain data from {self.data_dir}...")
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        json_files = list(self.data_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {self.data_dir}")
        
        print(f"Found {len(json_files)} value chain files")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    chain_data = json.load(f)
                
                self._process_value_chain(chain_data)
                
            except Exception as e:
                print(f"Warning: Error loading {json_file}: {e}")
                continue
        
        print(f"Loaded {len(self.documents)} documents")
        print(f"Company-Chain mappings: {len(self.company_to_chains)}")
        print(f"Chain-Company mappings: {len(self.chain_to_companies)}")
        
        # Debug: Show some sample company names
        sample_companies = list(self.company_to_chains.keys())[:10]
        print(f"Sample companies in KB: {sample_companies}")
        
        # Debug: Check if specific companies are in the KB
        test_companies = ["91APP*-KY", "91APP", "ACpay", "IET-KY", "台積電"]
        found_companies = [c for c in test_companies if c in self.company_to_chains]
        print(f"Test companies found: {found_companies}")
        
        if len(found_companies) > 0:
            # Show chains for first found company
            first_company = found_companies[0]
            chains_for_company = self.company_to_chains[first_company]
            print(f"Example: {first_company} belongs to {len(chains_for_company)} chains")
    
    def _process_value_chain(self, chain_data: Dict):
        """Process a single value chain and create documents."""
        try:
            chain_name = chain_data.get('title', '')
            introduction = chain_data.get('introduction', '')
            
            if not chain_name:
                return
            
            # Create main chain document
            main_doc_content = f"產業鏈名稱: {chain_name}\n"
            if introduction:
                # Truncate very long introductions
                intro_text = introduction[:1000] + "..." if len(introduction) > 1000 else introduction
                main_doc_content += f"介紹: {intro_text}\n"
            
            main_doc = Document(
                page_content=main_doc_content,
                metadata={
                    "type": "value_chain",
                    "chain_name": chain_name
                }
            )
            self.documents.append(main_doc)
            
            # Initialize chain company list
            if chain_name not in self.chain_to_companies:
                self.chain_to_companies[chain_name] = set()
            
            # Process chains structure (which contains categories and companies)
            chains = chain_data.get('chains', [])
            for chain_section in chains:
                self._process_chain_section(chain_section, chain_name)
            
        except Exception as e:
            print(f"Warning: Error processing chain data: {e}")
    
    def _process_chain_section(self, chain_section: Dict, chain_name: str):
        """Process a chain section (category) and extract companies."""
        try:
            section_title = chain_section.get('title', '')
            companies = chain_section.get('companies', [])
            
            # Create section document
            section_content = f"產業鏈: {chain_name}\n類別: {section_title}\n"
            
            # Process companies in this section
            companies_in_section = set()
            
            for company_item in companies:
                if isinstance(company_item, dict):
                    # Check if it's a detailed_data structure
                    if 'detailed_data' in company_item:
                        detailed_data = company_item['detailed_data']
                        sub_companies = detailed_data.get('companies', [])
                        
                        for company_info in sub_companies:
                            if isinstance(company_info, dict):
                                company_name = company_info.get('name', '')
                                is_foreign = company_info.get('is_foreign', False)
                                
                                if company_name:
                                    companies_in_section.add(company_name)
                                    self._add_company_mapping(company_name, chain_name, section_title, is_foreign)
                    
                    # Check if it's a direct company entry
                    elif 'name' in company_item:
                        company_name = company_item.get('name', '')
                        is_foreign = company_item.get('is_foreign', False)
                        
                        if company_name:
                            companies_in_section.add(company_name)
                            self._add_company_mapping(company_name, chain_name, section_title, is_foreign)
            
            # Add companies list to section content
            if companies_in_section:
                section_content += f"包含公司: {', '.join(sorted(companies_in_section))}\n"
            
            # Create section document
            section_doc = Document(
                page_content=section_content,
                metadata={
                    "type": "category",
                    "chain_name": chain_name,
                    "category": section_title,
                    "company_count": len(companies_in_section)
                }
            )
            self.documents.append(section_doc)
            
        except Exception as e:
            print(f"Warning: Error processing chain section: {e}")
    
    def _add_company_mapping(self, company_name: str, chain_name: str, category: str, is_foreign: bool):
        """Add company to chain mappings and create company document."""
        try:
            # Update mappings
            if company_name not in self.company_to_chains:
                self.company_to_chains[company_name] = set()
            self.company_to_chains[company_name].add(chain_name)
            self.chain_to_companies[chain_name].add(company_name)
            
            # Create company document
            comp_content = f"公司名稱: {company_name}\n"
            comp_content += f"產業鏈: {chain_name}\n"
            comp_content += f"類別: {category}\n"
            comp_content += f"外國公司: {'是' if is_foreign else '否'}\n"
            
            comp_doc = Document(
                page_content=comp_content,
                metadata={
                    "type": "company",
                    "company_name": company_name,
                    "chain_name": chain_name,
                    "category": category,
                    "is_foreign": is_foreign
                }
            )
            self.documents.append(comp_doc)
            
        except Exception as e:
            print(f"Warning: Error adding company mapping: {e}")


class RAGModelProvider:
    """Base class for RAG-enabled model providers."""
    
    def __init__(self, config: RAGConfig, knowledge_base: TaiwanValueChainKnowledgeBase):
        self.config = config
        self.knowledge_base = knowledge_base
        self.llm = self._initialize_model()
        self.embeddings = self._initialize_embeddings()
        self.vector_store = self._create_vector_store()
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={
                "k": config.top_k,
                "score_threshold": config.score_threshold
            }
        )
    
    def _initialize_model(self) -> BaseLanguageModel:
        """Initialize the language model."""
        if self.config.provider == 'openai':
            api_key = self.config.api_key or os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key required")
            
            return ChatOpenAI(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                api_key=api_key,
                request_timeout=self.config.timeout
            )
        
        elif self.config.provider == 'anthropic':
            api_key = self.config.api_key or os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("Anthropic API key required")
            
            return ChatAnthropic(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                anthropic_api_key=api_key,
                timeout=self.config.timeout
            )
        
        elif self.config.provider == 'google':
            api_key = self.config.api_key or os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("Google API key required")
            
            return ChatGoogleGenerativeAI(
                model=self.config.model_name,
                max_output_tokens=self.config.max_tokens,
                google_api_key=api_key
            )
        
        elif self.config.provider == 'ollama':
            base_url = self.config.base_url or "http://localhost:11434"
            extra_params = dict(self.config.extra_params or {}) 

            model_lower = (self.config.model_name or "").lower()

            # === reasoning 規則 ===
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
                    # 其他模型：直接關 reasoning
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
        
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    def _initialize_embeddings(self):
        """Initialize embedding model."""
        # Define default embedding models for each provider
        default_embedding_models = {
            "openai": "text-embedding-3-small",
            "huggingface": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "ollama": "nomic-embed-text",
            "google": "models/gemini-embedding-001"  # Google Gemini embedding model
        }
        
        # Use configured model or provider-specific default
        embedding_model = self.config.embedding_model or default_embedding_models.get(
            self.config.embedding_provider, 
            default_embedding_models["huggingface"]  # Fallback to free HuggingFace model
        )
        
        # Store the actual model being used for logging
        self.config.embedding_model = embedding_model
        
        if self.config.embedding_provider == "openai":
            api_key = self.config.api_key or os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key required for embeddings")
            
            return OpenAIEmbeddings(
                model=embedding_model,
                openai_api_key=api_key
            )
        
        elif self.config.embedding_provider == "huggingface":
            # Use configured model or default multilingual model
            return HuggingFaceEmbeddings(
                model_name=embedding_model
            )
        
        elif self.config.embedding_provider == "ollama":
            base_url = self.config.base_url or "http://localhost:11434"
            return OllamaEmbeddings(
                model=embedding_model,
                base_url=base_url
            )
        
        elif self.config.embedding_provider == "google":
            api_key = self.config.api_key or os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("Google API key required for embeddings")
            
            return GoogleGenerativeAIEmbeddings(
                model=embedding_model,
                google_api_key=api_key
            )
        
        else:
            raise ValueError(f"Unsupported embedding provider: {self.config.embedding_provider}")
    
    def _create_vector_store(self):
        """Create vector store from knowledge base."""
        print(f"Creating vector store with {len(self.knowledge_base.documents)} documents...")
        print(f"Using {self.config.embedding_provider} embeddings ({self.config.embedding_model})")
        
        if not self.knowledge_base.documents:
            raise ValueError("No documents in knowledge base")
        
        vector_store = FAISS.from_documents(
            self.knowledge_base.documents,
            self.embeddings
        )
        
        print("✓ Vector store created successfully")
        return vector_store
    
    async def retrieve_and_generate(self, question: str, dataset_type: str, company: Optional[str] = None, chains: Optional[List[str]] = None) -> Dict[str, Any]:
        """Retrieve relevant context and generate answer."""
        try:
            # Retrieve relevant documents using direct similarity search
            print(f"Retrieving documents for: {question[:50]}...")
            
            search_terms = []
            search_term = None

            # Extract search term based on dataset type
            if dataset_type == 'firm_chains_qa':
                # Extract company name from question for better search
                company_match = re.search(r'公司\s+([\w\*\-\+\(\)]+)', question)
                if company_match:
                    search_term = company_match.group(1)
                else:
                    search_term = question
            elif dataset_type == 'chain_firms_qa':
                # Extract chain name from question
                chain_match = re.search(r'產業鏈\s+([\w\*\-\+\(\)產業鏈]+)', question)
                if chain_match:
                    search_term = chain_match.group(1)
                    # Remove duplicate "產業鏈" if present
                    if search_term.endswith('產業鏈產業鏈'):
                        search_term = search_term[:-3]  # Remove one "產業鏈"
                else:
                    search_term = question
            else: # competitors_qa
                if company:
                    search_terms.append(company)
                if chains:
                    search_terms.extend(chains)
                if not search_terms:
                    search_terms = [question]
            if dataset_type in ("firm_chains_qa", "chain_firms_qa"):
                search_terms = [search_term]
            # Use similarity search directly (this works better than retriever)
            # retrieved_docs = self.vector_store.similarity_search(search_term, k=self.config.top_k)
            # Multi-query retrieval with score merge (better for competitors)
            scored = []
            for term in search_terms:
                try:
                    scored.extend(self.vector_store.similarity_search_with_score(term, k=self.config.top_k))
                except Exception:
                    # fallback if with_score not available
                    docs = self.vector_store.similarity_search(term, k=self.config.top_k)
                    scored.extend([(d, 0.0) for d in docs])

            # sort by score (FAISS: smaller distance can mean more similar depending on setup;
            # LangChain FAISS typically returns L2 distance. We'll still sort ascending.)
            scored.sort(key=lambda x: x[1])

            # de-dup by (page_content, metadata)
            seen = set()
            retrieved_docs = []
            for doc, score in scored:
                key = (doc.page_content, tuple(sorted(doc.metadata.items())))
                if key in seen:
                    continue
                seen.add(key)
                retrieved_docs.append(doc)
                if len(retrieved_docs) >= self.config.top_k:
                    break
            print(f"Retrieved {len(retrieved_docs)} documents")
            
            if len(retrieved_docs) == 0:
                print(f"No documents retrieved for search term: {search_term}")
            
            # Create context from retrieved documents
            context = "\n\n".join([
                f"[文件 {i+1}] {doc.page_content}" 
                for i, doc in enumerate(retrieved_docs)
            ])
            
            # Create RAG prompt
            if dataset_type == 'firm_chains_qa':
                system_instruction = """你是一位熟悉台灣產業鏈的專家。請根據提供的參考資料回答問題。

參考資料：
{context}

要求：
1. 只列出產業鏈的名稱，每個產業鏈一行
2. 不要包含編號、項目符號或其他格式
3. 產業鏈名稱應該精確，例如「半導體產業鏈」、「電動車產業鏈」等
4. 如果參考資料中沒有相關資訊，請回答「根據提供的資料無法確定」
5. 不要編造或猜測不在參考資料中的產業鏈"""
            
            elif dataset_type == 'chain_firms_qa':
                system_instruction = """你是一位熟悉台灣產業鏈的專家。請根據提供的參考資料回答問題。

參考資料：
{context}

要求：
1. 只列出公司名稱，每個公司一行
2. 不要包含編號、項目符號或其他格式
3. 公司名稱應該精確，包含台灣本地公司和外國公司
4. 如果參考資料中沒有相關資訊，請回答「根據提供的資料無法確定」
5. 不要編造或猜測不在參考資料中的公司名稱"""
            else:  # competitors_qa
                system_instruction = """你是一位熟悉台灣產業鏈的專家。請根據提供的參考資料回答問題。

參考資料：
{context}

要求：
1. 只列出「競爭對手公司名稱」，每個公司一行
2. 不要包含編號、項目符號或其他格式
3. 公司名稱應該精確，包含台灣本地公司和外國公司
4. 若參考資料不足以判斷，請回答「根據提供的資料無法確定」
5. 不要編造或猜測不在參考資料中的公司名稱
"""
            
            # Create messages - use both SystemMessage and HumanMessage for Google Gemini compatibility
            system_text = system_instruction.format(context=context)
            messages = [
                SystemMessage(content=system_text),
                HumanMessage(content=question)
            ]
            
            print(f"Sending prompt to {self.config.provider} {self.config.model_name}...")
            
            # Generate response
            response = await self.llm.ainvoke(messages)
            answer = response.content.strip()
            
            print(f"Received response: {answer[:100]}...")
            
            return {
                "question": question,
                "answer": answer,
                "context": context,
                "retrieved_docs_count": len(retrieved_docs),
                "retrieved_docs": [
                    {
                        "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in retrieved_docs
                ]
            }
        
        except Exception as e:
            print(f"ERROR in retrieve_and_generate: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "question": question,
                "answer": None,
                "error": str(e),
                "context": "",
                "retrieved_docs_count": 0,
                "retrieved_docs": []
            }


class RAGEvaluator:
    """RAG evaluator for Taiwan value chain QA datasets."""
    
    def __init__(self, config: RAGConfig):
        """Initialize RAG evaluator."""
        self.config = config
        self.knowledge_base = TaiwanValueChainKnowledgeBase(data_dir=self.config.data_dir)
        self.provider = RAGModelProvider(config, self.knowledge_base)
        self.results = []
        self.dataset_type = None
    
    def detect_dataset_type(self, dataset_sample: Dict) -> str:
        """Detect whether this is firm_chains_qa or chain_firms_qa dataset."""
        if 'company' in dataset_sample and 'chains' in dataset_sample:
            return 'competitors_qa'
        elif 'company' in dataset_sample:
            return 'firm_chains_qa'
        elif 'chain' in dataset_sample:
            return 'chain_firms_qa'
        else:
            raise ValueError("Unknown dataset format. Expected 'company' or 'chain' field in data.")
    
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
        
        # Split by newlines and clean up
        lines = response.split('\n')
        items = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Remove common prefixes (numbers, bullets, dashes)
            line = re.sub(r'^[\d\.\-\*\•\→]+\s*', '', line)
            line = re.sub(r'^\s*[-·]\s*', '', line)
            
            # Remove trailing punctuation
            line = re.sub(r'[，。、；：]$', '', line)
            
            # Skip if too short
            if len(line) < 2:
                continue
            
            if dataset_type == 'firm_chains_qa':
                # Extract chain name if it contains "產業鏈"
                if '產業鏈' in line:
                    items.append(line.strip())
            elif dataset_type in ('chain_firms_qa', 'competitors_qa'):
                # For companies, accept any non-empty string with reasonable content
                if re.search(r'[\u4e00-\u9fff]', line) or re.search(r'[A-Za-z]', line):
                    items.append(line.strip())
        
        return items
    
    def calculate_metrics(self, predicted: List[str], actual: List[str]) -> Dict[str, float]:
        """Calculate evaluation metrics for a single prediction."""
        # Convert to sets for comparison
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
            'actual_count': len(actual_set)
        }
    
    async def evaluate_dataset(self, dataset_path: str, max_samples: Optional[int] = None,
                              sample_rate: float = 1.0, save_results: bool = True) -> Dict:
        """Evaluate RAG performance on the dataset."""
        
        # Load dataset and detect type
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f]
        
        # Auto-detect dataset type from first sample
        self.dataset_type = self.detect_dataset_type(dataset[0])
        
        print(f"\n{'='*70}")
        print(f"Evaluating RAG with {self.config.provider.upper()} {self.config.model_name}")
        print(f"Dataset Type: {self.dataset_type}")
        print(f"{'='*70}\n")
        print(f"Dataset: {dataset_path}")
        print(f"Provider: {self.config.provider}")
        print(f"Model: {self.config.model_name}")
        print(f"Embedding: {self.config.embedding_provider} ({self.config.embedding_model})")
        print(f"RAG Config: top_k={self.config.top_k}, threshold={self.config.score_threshold}")
        
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
        print(f"\nStarting RAG evaluation...\n")
        
        # Evaluation metrics
        results = []
        total_recall = 0.0
        total_precision = 0.0
        total_f1 = 0.0
        total_exact_match = 0
        
        # Track errors
        error_analysis = {
            'api_errors': 0,
            'empty_responses': 0,
            'retrieval_errors': 0
        }
        
        start_time = time.time()
        
        for idx, item in enumerate(dataset, 1):
            question = item['question']
            actual_answer = item['answer']
            answer_count = item['answer_count']
            
            if self.dataset_type == 'firm_chains_qa':
                entity = item['company']
                is_foreign = item['is_foreign']
                print(f"[{idx}/{len(dataset)}] {entity} ({answer_count} chains)...", end=' ')
            elif self.dataset_type == 'chain_firms_qa':
                entity = item['chain']
                local_count = item.get('local_count', 0)
                foreign_count = item.get('foreign_count', 0)
                print(f"[{idx}/{len(dataset)}] {entity} ({answer_count} companies)...", end=' ')
            else :  # competitors_qa
                entity = item['company']
                chains = item.get('chains', [])
                local_count = item.get('local_count', 0)
                foreign_count = item.get('foreign_count', 0)
                print(f"[{idx}/{len(dataset)}] {entity} ({answer_count} competitors, {len(chains)} chains)...", end=' ')

            # RAG query
            if self.dataset_type == 'competitors_qa':
                rag_result = await self.provider.retrieve_and_generate(
                    question,
                    self.dataset_type,
                    company=item.get('company'),
                    chains=item.get('chains', [])
                )
            else:
                rag_result = await self.provider.retrieve_and_generate(question, self.dataset_type)

            if rag_result.get('error'):
                print("❌ RAG Error")
                error_analysis['api_errors'] += 1
                predicted_answer = []
                response = None
            else:
                response = rag_result['answer']
                predicted_answer = self.parse_response(response, self.dataset_type)
                
                if not predicted_answer:
                    error_analysis['empty_responses'] += 1
                
                print(f"✓ ({len(predicted_answer)} predicted, {rag_result['retrieved_docs_count']} docs)")
            
            # Calculate metrics
            metrics = self.calculate_metrics(predicted_answer, actual_answer)
            
            # Store result
            result = {
                'index': idx,
                'entity': entity,
                'question': question,
                'actual_answer': actual_answer,
                'predicted_answer': predicted_answer,
                'response': response,
                'rag_context': rag_result.get('context', ''),
                'retrieved_docs_count': rag_result.get('retrieved_docs_count', 0),
                'retrieved_docs': rag_result.get('retrieved_docs', []),
                'metrics': metrics,
                'dataset_type': self.dataset_type
            }
            
            # Add dataset-specific fields
            if self.dataset_type == 'firm_chains_qa':
                result['company'] = entity
                result['is_foreign'] = is_foreign
            elif self.dataset_type == 'chain_firms_qa':
                result['chain'] = entity
                result['local_count'] = local_count
                result['foreign_count'] = foreign_count
            else:  # competitors_qa
                result['company'] = item.get('company')
                result['chains'] = item.get('chains', [])
                result['local_count'] = item.get('local_count', 0)
                result['foreign_count'] = item.get('foreign_count', 0)
                result['is_foreign'] = item.get('is_foreign', False)
            
            results.append(result)
            
            # Aggregate metrics
            total_recall += metrics['recall']
            total_precision += metrics['precision']
            total_f1 += metrics['f1']
            if metrics['exact_match'] == 1.0:
                total_exact_match += 1
            
            # Rate limiting for API providers
            if self.config.provider != 'ollama':
                await asyncio.sleep(0.1)
        
        elapsed_time = time.time() - start_time
        
        # Calculate average metrics
        n = len(dataset)
        avg_metrics = {
            'recall': total_recall / n,
            'precision': total_precision / n,
            'f1': total_f1 / n,
            'exact_match_rate': total_exact_match / n,
            'evaluated_samples': n,
            'total_samples': total_samples,
            'elapsed_time': elapsed_time,
            'avg_time_per_sample': elapsed_time / n
        }
        
        # Compile full results
        full_results = {
            'provider': self.config.provider,
            'model': self.config.model_name,
            'embedding_provider': self.config.embedding_provider,
            'embedding_model': self.config.embedding_model,
            'rag_config': {
                'top_k': self.config.top_k,
                'score_threshold': self.config.score_threshold
            },
            'dataset': str(dataset_path),
            'dataset_type': self.dataset_type,
            'timestamp': datetime.now().isoformat(),
            'average_metrics': avg_metrics,
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.config.model_name.replace("/", "_").replace(":", "_")
        
        # Save to top-level results/ directory
        workspace_root = Path(__file__).parent.parent.parent
        output_dir = workspace_root / 'results'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"rag_evaluation_results_{dataset_name}_{self.config.provider}_{model_name}_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Detailed results saved to: {output_file}")
    
    def _print_summary(self, results: Dict):
        """Print evaluation summary."""
        metrics = results['average_metrics']
        errors = results['error_analysis']
        rag_config = results['rag_config']
        
        print(f"\n{'='*70}")
        print("RAG EVALUATION SUMMARY")
        print(f"{'='*70}\n")
        
        print(f"Provider: {results['provider']}")
        print(f"Model: {results['model']}")
        print(f"Embedding: {results['embedding_provider']} ({results['embedding_model']})")
        print(f"RAG Config: top_k={rag_config['top_k']}, threshold={rag_config['score_threshold']}")
        print(f"Dataset Type: {results['dataset_type']}")
        print(f"Evaluated: {metrics['evaluated_samples']} samples")
        print(f"Time: {metrics['elapsed_time']:.1f}s ({metrics['avg_time_per_sample']:.2f}s per sample)")
        
        print(f"\n{'='*70}")
        print("RAG PERFORMANCE METRICS")
        print(f"{'='*70}")
        print(f"  Recall:            {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"  Precision:         {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"  F1 Score:          {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
        print(f"  Exact Match Rate:  {metrics['exact_match_rate']:.4f} ({metrics['exact_match_rate']*100:.2f}%)")
        
        print(f"\n{'='*70}")
        print("ERROR ANALYSIS")
        print(f"{'='*70}")
        print(f"  API Errors:        {errors['api_errors']}")
        print(f"  Empty Responses:   {errors['empty_responses']}")
        print(f"  Retrieval Errors:  {errors['retrieval_errors']}")
        
        # Show some example predictions
        detailed = results['detailed_results']
        
        # Show perfect predictions
        perfect = [r for r in detailed if r['metrics']['exact_match'] == 1.0]
        if perfect:
            print(f"\n{'='*70}")
            print(f"PERFECT PREDICTIONS (showing first 3 of {len(perfect)})")
            print(f"{'='*70}")
            for r in perfect[:3]:
                entity_name = r.get('company', r.get('chain', r['entity']))
                predicted = r['predicted_answer']
                print(f"\n  Entity: {entity_name}")
                print(f"  Predicted: {predicted}")
                print(f"  Retrieved docs: {r['retrieved_docs_count']}")
        
        # Show partial matches
        partial = [r for r in detailed if 0 < r['metrics']['recall'] < 1.0]
        if partial:
            print(f"\n{'='*70}")
            print(f"PARTIAL MATCHES (showing first 3 of {len(partial)})")
            print(f"{'='*70}")
            for r in partial[:3]:
                entity_name = r.get('company', r.get('chain', r['entity']))
                actual = r['actual_answer']
                predicted = r['predicted_answer']
                print(f"\n  Entity: {entity_name}")
                print(f"  Actual:    {actual}")
                print(f"  Predicted: {predicted}")
                print(f"  Recall: {r['metrics']['recall']:.2f}, Precision: {r['metrics']['precision']:.2f}")
                print(f"  Retrieved docs: {r['retrieved_docs_count']}")
        
        print(f"\n{'='*70}\n")


async def main():
    parser = argparse.ArgumentParser(
        description='RAG Evaluation for Taiwan Value Chain QA datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # RAG with OpenAI
  python evaluate_rag_models.py --dataset datasets/qa/firm_chains_qa_local.jsonl --provider openai --model gpt-4o-mini
  
  # RAG with Ollama
  python evaluate_rag_models.py --dataset datasets/qa/firm_chains_qa_local.jsonl --provider ollama --model gemma3:4b
  
  # Custom RAG parameters
  python evaluate_rag_models.py --dataset datasets/qa/firm_chains_qa_local.jsonl --provider openai --model gpt-4o-mini --top-k 10 --score-threshold 0.5
  
  # Use HuggingFace embeddings (free)
  python evaluate_rag_models.py --dataset datasets/qa/firm_chains_qa_local.jsonl --provider ollama --model gemma3:4b --embedding-provider huggingface
        """
    )
    
    parser.add_argument('--dataset', '-d', required=True,
                       help='Path to JSONL dataset file')
    parser.add_argument('--provider', '-p', required=True,
                       choices=['openai', 'anthropic', 'google', 'ollama'],
                       help='Model provider')
    parser.add_argument('--model', '-m', required=True,
                       help='Model name')
    parser.add_argument('--api-key', '-k',
                       help='API key for the provider')
    parser.add_argument('--base-url',
                       help='Base URL for API (useful for Ollama)')
    parser.add_argument('--temperature', '-t', type=float, default=0.0,
                       help='Temperature for generation (default: 0.0)')
    parser.add_argument('--max-tokens', type=int, default=500,
                       help='Maximum tokens for generation (default: 500)')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Request timeout in seconds (default: 30)')
    
    # RAG-specific arguments
    parser.add_argument('--embedding-provider', choices=['openai', 'huggingface', 'ollama', 'google'], 
                       default='openai', help='Embedding model provider (default: openai). Note: Google has batch limit ~100 requests')
    parser.add_argument('--embedding-model', default=None,
                       help='Embedding model name (default: provider-specific, e.g., text-embedding-3-small for openai, models/gemini-embedding-001 for google, nomic-embed-text for ollama)')
    # resoning switch
    parser.add_argument(
        '--no-reasoning',
        action='store_true',
        help='Disable reasoning for Ollama models (default: reasoning enabled)'
    )
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of documents to retrieve (default: 5)')
    parser.add_argument('--score-threshold', type=float, default=0.0,
                       help='Minimum similarity score for retrieval (default: 0.0)')
    
    parser.add_argument('--max-samples', type=int,
                       help='Maximum number of samples to evaluate (default: all)')
    parser.add_argument('--sample-rate', type=float, default=1.0,
                       help='Rate of samples to evaluate, 0.0-1.0 (default: 1.0)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save detailed results to file')
    parser.add_argument('--data-dir', type=str,
                       default='datasets/demo/individual_chains',
                       help='Path to directory containing individual chain JSON files (default: datasets/demo/individual_chains)')
    
    args = parser.parse_args()
    
    # Validate dataset path
    if not Path(args.dataset).exists():
        print(f"✗ Error: Dataset file not found: {args.dataset}")
        exit(1)
    
    # Validate sample rate
    if not 0.0 < args.sample_rate <= 1.0:
        print("✗ Error: Sample rate must be between 0.0 and 1.0")
        exit(1)
    
    # Create RAG configuration
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
        enable_reasoning=(not args.no_reasoning),
    )
    
    # Create RAG evaluator
    try:
        evaluator = RAGEvaluator(config)
    except Exception as e:
        print(f"✗ Error initializing RAG evaluator: {e}")
        exit(1)
    
    # Run evaluation
    try:
        results = await evaluator.evaluate_dataset(
            args.dataset,
            max_samples=args.max_samples,
            sample_rate=args.sample_rate,
            save_results=not args.no_save
        )
        
        print("✓ RAG evaluation completed successfully!")
        
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