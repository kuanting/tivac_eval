"""
LLM Provider implementations for the evaluation system.
"""

import os
from typing import List, Dict, Optional, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.base import BaseLanguageModel

from config import ModelConfig


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
    
    def _convert_messages(self, messages: List[Dict[str, str]]) -> List:
        """Convert dict messages to LangChain message objects."""
        lc_messages = []
        for msg in messages:
            if msg["role"] == "system":
                lc_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
        return lc_messages


class OpenAIProvider(ModelProvider):
    """OpenAI model provider."""
    
    def _initialize_model(self):
        from langchain_openai import ChatOpenAI
        
        api_key = self.config.api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass via --api-key")
        
        extra_params = self.config.extra_params or {}
        
        return ChatOpenAI(
            model=self.config.model_name,
            max_tokens=self.config.max_tokens,
            api_key=api_key,
            request_timeout=self.config.timeout,
            **extra_params
        )
    
    async def generate(self, messages: List[Dict[str, str]]) -> Optional[str]:
        try:
            lc_messages = self._convert_messages(messages)
            response = await self.model.ainvoke(lc_messages)
            return response.content.strip()
        except Exception as e:
            print(f"OpenAI API Error: {e}")
            return None


class AnthropicProvider(ModelProvider):
    """Anthropic Claude model provider."""
    
    def _initialize_model(self):
        from langchain_anthropic import ChatAnthropic
        
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
    
    async def generate(self, messages: List[Dict[str, str]]) -> Optional[str]:
        try:
            lc_messages = self._convert_messages(messages)
            response = await self.model.ainvoke(lc_messages)
            return response.content.strip()
        except Exception as e:
            print(f"Anthropic API Error: {e}")
            return None


class GoogleProvider(ModelProvider):
    """Google Gemini model provider."""
    
    def _initialize_model(self):
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        api_key = self.config.api_key or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("Google API key required. Set GOOGLE_API_KEY environment variable or pass via --api-key")
        
        extra_params = self.config.extra_params or {}
        
        return ChatGoogleGenerativeAI(
            model=self.config.model_name,
            max_output_tokens=self.config.max_tokens,
            google_api_key=api_key,
            **extra_params
        )
    
    async def generate(self, messages: List[Dict[str, str]]) -> Optional[str]:
        try:
            lc_messages = self._convert_messages(messages)
            response = await self.model.ainvoke(lc_messages)
            return response.content.strip()
        except Exception as e:
            print(f"Google API Error: {e}")
            return None


class OllamaProvider(ModelProvider):
    """Ollama local model provider."""
    
    def _initialize_model(self):
        from langchain_ollama import ChatOllama
        
        base_url = self.config.base_url or "http://localhost:11434"
        extra_params = self.config.extra_params or {}

        model_lower = self.config.model_name.lower()
        extra_params["think"] = "low" if "gpt" in model_lower else False
        
        return ChatOllama(
            model=self.config.model_name,
            base_url=base_url,
            **extra_params
        )
    
    async def generate(self, messages: List[Dict[str, str]]) -> Optional[str]:
        try:
            lc_messages = self._convert_messages(messages)
            response = await self.model.ainvoke(lc_messages)
            return response.content.strip()
        except Exception as e:
            print(f"Ollama Error: {e}")
            return None


# Provider registry
PROVIDERS = {
    'openai': {
        'class': OpenAIProvider,
        'models': ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo'],
        'env_key': 'OPENAI_API_KEY'
    },
    'anthropic': {
        'class': AnthropicProvider,
        'models': ['claude-3-5-sonnet-20241022', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'],
        'env_key': 'ANTHROPIC_API_KEY'
    },
    'google': {
        'class': GoogleProvider,
        'models': ['gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-pro'],
        'env_key': 'GOOGLE_API_KEY'
    },
    'ollama': {
        'class': OllamaProvider,
        'models': ['llama3.1', 'llama2', 'mistral', 'codellama', 'qwen2', 'gemma2', 'gemma3:2b', 'gemma3:4b'],
        'env_key': None
    }
}


def get_provider(config: ModelConfig) -> ModelProvider:
    """Get the appropriate provider for a model configuration.
    
    Args:
        config: ModelConfig with provider and model information
        
    Returns:
        Initialized ModelProvider instance
        
    Raises:
        ValueError: If provider is not supported
    """
    if config.provider not in PROVIDERS:
        raise ValueError(f"Unsupported provider: {config.provider}. Available: {list(PROVIDERS.keys())}")
    
    provider_class = PROVIDERS[config.provider]['class']
    return provider_class(config)


def list_available_providers() -> Dict[str, Dict[str, Any]]:
    """List all available providers and their models."""
    return {
        provider: {
            'models': info['models'],
            'requires_api_key': info['env_key'] is not None,
            'env_key': info['env_key']
        }
        for provider, info in PROVIDERS.items()
    }
