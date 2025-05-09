import os
import time
from typing import Any, Dict, List, Optional, Union

# Import the required client libraries
try:
    import openai
except ImportError:
    openai = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from transformers import pipeline
    import torch
except ImportError:
    pipeline = None
    torch = None

from hackaikit.core.base_module import BaseModule


class LLMToolsModule(BaseModule):
    """
    LLM Tools Module for HackAI-Kit.
    
    This module provides integrations with various Large Language Model providers
    including OpenAI (ChatGPT), Google (Gemini), and Hugging Face.
    
    Supported tasks:
    - chat: Conversational interactions
    - text_generation: Generate text from prompts
    - summarization: Summarize long texts
    - question_answering: Answer questions based on context
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the LLM Tools module.
        
        Args:
            config_manager: Configuration manager to access API keys
        """
        super().__init__(config_manager)
        self.providers = {
            "openai": self._setup_openai,
            "gemini": self._setup_gemini,
            "huggingface": self._setup_huggingface
        }
        self.models = {}
        self.huggingface_pipelines = {}
        
        # Set up clients if API keys are available
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize API clients based on available API keys."""
        if self.config_manager:
            # Set up OpenAI if API key is available
            openai_api_key = self.config_manager.get_api_key("openai")
            if openai_api_key and openai:
                self._setup_openai(openai_api_key)
            
            # Set up Gemini if API key is available
            gemini_api_key = self.config_manager.get_api_key("google")
            if gemini_api_key and genai:
                self._setup_gemini(gemini_api_key)
            
            # Set up Hugging Face if API key is available
            hf_api_key = self.config_manager.get_api_key("huggingface")
            if hf_api_key and pipeline:
                self._setup_huggingface(hf_api_key)
    
    def _setup_openai(self, api_key):
        """
        Set up the OpenAI client.
        
        Args:
            api_key: OpenAI API key
        """
        if not openai:
            print("Warning: OpenAI package not installed. Run 'pip install openai'")
            return False
            
        try:
            openai.api_key = api_key
            # Test connection with a simple completion
            openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            print("OpenAI client initialized successfully.")
            return True
        except Exception as e:
            print(f"Error initializing OpenAI client: {str(e)}")
            return False
    
    def _setup_gemini(self, api_key):
        """
        Set up the Google Gemini client.
        
        Args:
            api_key: Google API key
        """
        if not genai:
            print("Warning: Google Generative AI package not installed. Run 'pip install google-generativeai'")
            return False
            
        try:
            genai.configure(api_key=api_key)
            # Test connection
            model = genai.GenerativeModel('gemini-pro')
            model.generate_content("Hello")
            print("Gemini client initialized successfully.")
            return True
        except Exception as e:
            print(f"Error initializing Gemini client: {str(e)}")
            return False
    
    def _setup_huggingface(self, api_key):
        """
        Set up Hugging Face access.
        
        Args:
            api_key: Hugging Face API token
        """
        if not pipeline:
            print("Warning: Hugging Face Transformers package not installed. Run 'pip install transformers'")
            return False
            
        try:
            os.environ["HUGGINGFACE_TOKEN"] = api_key
            # We'll initialize pipelines on demand
            print("Hugging Face access initialized successfully.")
            return True
        except Exception as e:
            print(f"Error initializing Hugging Face access: {str(e)}")
            return False
    
    def get_supported_tasks(self) -> List[str]:
        """
        Get a list of tasks supported by the module.
        
        Returns:
            List of supported task names
        """
        return ["chat", "text_generation", "summarization", "question_answering"]
    
    def get_supported_algorithms(self) -> Dict[str, List[str]]:
        """
        Get a dictionary of supported algorithms for each task.
        
        Returns:
            Dictionary mapping task names to lists of algorithm names
        """
        return {
            "chat": ["openai", "gemini", "huggingface"],
            "text_generation": ["openai", "gemini", "huggingface"],
            "summarization": ["openai", "gemini", "huggingface"],
            "question_answering": ["openai", "gemini", "huggingface"]
        }
    
    def train(self, data, **kwargs):
        """
        Train method is not applicable for this module.
        
        For fine-tuning LLMs, use specialized methods specific to each provider.
        """
        raise NotImplementedError("Training LLMs requires specialized approaches not covered by this method.")
    
    def predict(self, model_id: str, data: Any, **kwargs) -> Dict[str, Any]:
        """
        Make predictions using an LLM.
        
        This is a wrapper around process() for consistency with other modules.
        
        Args:
            model_id: Not used for this module
            data: Input text
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with prediction results
        """
        provider = kwargs.pop("provider", "openai")
        model = kwargs.pop("model", None)
        task = kwargs.pop("task", "text_generation")
        
        return self.process(data=data, task=task, provider=provider, model=model, **kwargs)
    
    def evaluate(self, model_id: str, data=None, **kwargs):
        """
        Evaluate method is not applicable for standard LLM use.
        """
        raise NotImplementedError("Standard evaluation is not implemented for LLMs.")
    
    def visualize(self, **kwargs):
        """
        Visualization is not applicable for this module.
        """
        raise NotImplementedError("Visualization is not implemented for LLM tools.")
    
    def process(self, data: Any, task: str, provider: str = "openai", model: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Process data with the specified LLM provider.
        
        Args:
            data: Input data (text string or messages list)
            task: Task to perform (chat, text_generation, summarization, question_answering)
            provider: LLM provider (openai, gemini, huggingface)
            model: Model name (if None, uses provider default)
            **kwargs: Additional task-specific parameters
            
        Returns:
            Dictionary with processing results
        """
        # Validate task
        if task not in self.get_supported_tasks():
            return f"Error: Task '{task}' not supported. Supported tasks: {self.get_supported_tasks()}"
        
        # Validate provider
        if provider not in self.providers:
            return f"Error: Provider '{provider}' not supported. Supported providers: {list(self.providers.keys())}"
        
        # Get default model if none specified
        if model is None and self.config_manager:
            model = self.config_manager.get_default_model(provider)
        
        # Provider-specific processing
        if provider == "openai":
            return self._process_openai(data, task, model, **kwargs)
        elif provider == "gemini":
            return self._process_gemini(data, task, model, **kwargs)
        elif provider == "huggingface":
            return self._process_huggingface(data, task, model, **kwargs)
        else:
            return f"Error: Unknown provider '{provider}'"
    
    def _process_openai(self, data: Any, task: str, model: str = "gpt-3.5-turbo", **kwargs) -> Dict[str, Any]:
        """
        Process data using OpenAI models.
        
        Args:
            data: Input data
            task: Task to perform
            model: OpenAI model name
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with results
        """
        if not openai:
            return "Error: OpenAI package not installed"
            
        if openai.api_key is None:
            return "Error: OpenAI API key not set"
        
        try:
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", 500)
            
            if task == "chat":
                # Format messages if they're not already in the right format
                if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                    messages = data
                else:
                    return "Error: Chat requires a list of message dictionaries with 'role' and 'content'"
                
                response = openai.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                return {
                    "response": response.choices[0].message.content,
                    "model": model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                }
                
            elif task == "text_generation":
                # Convert to chat format for text generation
                prompt = data if isinstance(data, str) else str(data)
                messages = [{"role": "user", "content": prompt}]
                
                response = openai.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                return {
                    "generated_text": response.choices[0].message.content,
                    "model": model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                }
                
            elif task == "summarization":
                text = data if isinstance(data, str) else str(data)
                prompt = f"Please summarize the following text:\n\n{text}"
                messages = [{"role": "user", "content": prompt}]
                
                response = openai.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                return {
                    "summary": response.choices[0].message.content,
                    "model": model
                }
                
            elif task == "question_answering":
                if isinstance(data, dict) and "question" in data and "context" in data:
                    question = data["question"]
                    context = data["context"]
                    prompt = f"Context: {context}\n\nQuestion: {question}"
                    messages = [{"role": "user", "content": prompt}]
                else:
                    return "Error: Question answering requires a dict with 'question' and 'context' keys"
                
                response = openai.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                return {
                    "answer": response.choices[0].message.content,
                    "model": model
                }
            
            else:
                return f"Error: Unsupported task '{task}' for OpenAI"
                
        except Exception as e:
            return f"Error processing with OpenAI: {str(e)}"
    
    def _process_gemini(self, data: Any, task: str, model: str = "gemini-pro", **kwargs) -> Dict[str, Any]:
        """
        Process data using Google Gemini models.
        
        Args:
            data: Input data
            task: Task to perform
            model: Gemini model name
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with results
        """
        if not genai:
            return "Error: Google Generative AI package not installed"
        
        try:
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens")
            generation_config = {
                "temperature": temperature,
            }
            if max_tokens:
                generation_config["max_output_tokens"] = max_tokens
            
            gemini_model = genai.GenerativeModel(model)
            
            if task == "chat":
                # Format messages for Gemini
                if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                    chat = gemini_model.start_chat(history=[])
                    
                    # Process each message
                    for message in data:
                        role = message.get("role", "").lower()
                        content = message.get("content", "")
                        
                        if role == "user":
                            response = chat.send_message(content, generation_config=generation_config)
                        # For system and assistant messages, we'll skip (Gemini doesn't have direct equivalents)
                    
                    # Get the last response
                    response_text = chat.last.text if chat.last else ""
                    
                    return {
                        "response": response_text,
                        "model": model
                    }
                else:
                    return "Error: Chat requires a list of message dictionaries with 'role' and 'content'"
                
            elif task == "text_generation":
                prompt = data if isinstance(data, str) else str(data)
                response = gemini_model.generate_content(prompt, generation_config=generation_config)
                
                return {
                    "generated_text": response.text,
                    "model": model
                }
                
            elif task == "summarization":
                text = data if isinstance(data, str) else str(data)
                prompt = f"Please summarize the following text:\n\n{text}"
                response = gemini_model.generate_content(prompt, generation_config=generation_config)
                
                return {
                    "summary": response.text,
                    "model": model
                }
                
            elif task == "question_answering":
                if isinstance(data, dict) and "question" in data and "context" in data:
                    question = data["question"]
                    context = data["context"]
                    prompt = f"Context: {context}\n\nQuestion: {question}"
                    response = gemini_model.generate_content(prompt, generation_config=generation_config)
                else:
                    return "Error: Question answering requires a dict with 'question' and 'context' keys"
                
                return {
                    "answer": response.text,
                    "model": model
                }
            
            else:
                return f"Error: Unsupported task '{task}' for Gemini"
                
        except Exception as e:
            return f"Error processing with Gemini: {str(e)}"
    
    def _process_huggingface(self, data: Any, task: str, model: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Process data using Hugging Face models.
        
        Args:
            data: Input data
            task: Task to perform
            model: Hugging Face model name (if None, uses task default)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with results
        """
        if not pipeline:
            return "Error: Hugging Face Transformers package not installed"
        
        try:
            # Set task-specific default models
            task_defaults = {
                "text_generation": "gpt2",
                "summarization": "facebook/bart-large-cnn",
                "question_answering": "deepset/roberta-base-squad2",
                "chat": "facebook/blenderbot-400M-distill"
            }
            
            # Use specified model or default
            hf_model = model or task_defaults.get(task)
            if not hf_model:
                return f"Error: No default model for task '{task}'"
                
            # Get or create pipeline
            pipeline_key = f"{task}_{hf_model}"
            if pipeline_key not in self.huggingface_pipelines:
                if task == "chat":
                    # For chat, use conversational pipeline
                    self.huggingface_pipelines[pipeline_key] = pipeline("conversational", model=hf_model)
                else:
                    self.huggingface_pipelines[pipeline_key] = pipeline(task, model=hf_model)
            
            pipe = self.huggingface_pipelines[pipeline_key]
            
            if task == "text_generation":
                prompt = data if isinstance(data, str) else str(data)
                max_length = kwargs.get("max_tokens", 50)
                temperature = kwargs.get("temperature", 0.7)
                
                # Temperature in HF is controlled via top_k and top_p
                top_p = 0.9 if temperature > 0.7 else 0.6
                
                result = pipe(prompt, max_length=max_length, top_p=top_p, do_sample=True)
                
                if isinstance(result, list):
                    generated_text = result[0]["generated_text"]
                else:
                    generated_text = result["generated_text"]
                
                return {
                    "generated_text": generated_text,
                    "model": hf_model
                }
                
            elif task == "summarization":
                text = data if isinstance(data, str) else str(data)
                max_length = kwargs.get("max_tokens", 130)
                min_length = kwargs.get("min_length", 30)
                
                result = pipe(text, max_length=max_length, min_length=min_length)
                summary = result[0]["summary_text"]
                
                return {
                    "summary": summary,
                    "model": hf_model
                }
                
            elif task == "question_answering":
                if isinstance(data, dict) and "question" in data and "context" in data:
                    question = data["question"]
                    context = data["context"]
                    
                    result = pipe(question=question, context=context)
                    
                    return {
                        "answer": result["answer"],
                        "score": result["score"],
                        "model": hf_model
                    }
                else:
                    return "Error: Question answering requires a dict with 'question' and 'context' keys"
                    
            elif task == "chat":
                if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                    from transformers import Conversation
                    
                    # Create a new conversation
                    conversation = Conversation()
                    
                    # Add user/bot messages to the conversation
                    for message in data:
                        role = message.get("role", "").lower()
                        content = message.get("content", "")
                        
                        if role == "user":
                            conversation.add_user_input(content)
                        elif role in ["assistant", "bot"]:
                            conversation.append_response(content)
                    
                    # Generate a response
                    result = pipe(conversation)
                    response = result.generated_responses[-1] if result.generated_responses else ""
                    
                    return {
                        "response": response,
                        "model": hf_model
                    }
                else:
                    return "Error: Chat requires a list of message dictionaries with 'role' and 'content'"
            
            else:
                return f"Error: Unsupported task '{task}' for Hugging Face"
                
        except Exception as e:
            return f"Error processing with Hugging Face: {str(e)}" 