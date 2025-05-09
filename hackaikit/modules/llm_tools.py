from hackaikit.core.base_module import BaseModule
from hackaikit.integrations.huggingface_utils import get_hf_pipeline
from hackaikit.integrations.gemini_utils import GeminiUtil
from hackaikit.integrations.openai_utils import OpenAIUtil
import os
import json
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Optional
import logging

logger = logging.getLogger(__name__)

class LLMToolsModule(BaseModule):
    """
    Module for using LLM-powered tools including text generation, question answering,
    text classification, summarization, and more.
    Supports OpenAI, Hugging Face, and Gemini.
    """
    def __init__(self, config_manager=None):
        super().__init__(config_manager)
        self.openai_util = None
        self.gemini_util = None
        self.hf_chat_pipeline = None
        self.hf_qa_pipeline = None
        self.hf_summarization_pipeline = None
        self.hf_classification_pipeline = None
        
        if self.config_manager:
            # Initialize OpenAI
            openai_key = self.config_manager.get_openai_key()
            if openai_key:
                self.openai_util = OpenAIUtil(api_key=openai_key)
            else:
                logger.warning("OpenAI API key not found. OpenAI features will be unavailable.")
                
            # Initialize Gemini
            gemini_key = self.config_manager.get_gemini_key()
            if gemini_key:
                self.gemini_util = GeminiUtil(api_key=gemini_key)
            else:
                logger.warning("Gemini API key not found. Gemini features will be unavailable.")
                
            # For Hugging Face, we'll initialize pipelines on-demand
        else:
            logger.warning("ConfigManager not provided. API integrations will be limited.")
    
    def process(self, data, task="chat", **kwargs):
        """Main processing method that routes to appropriate task"""
        if task == "chat":
            return self.chat(data, **kwargs)
        elif task == "question_answering":
            return self.question_answering(data, **kwargs)
        elif task == "summarization":
            return self.summarize(data, **kwargs)
        elif task == "classification":
            return self.classify(data, **kwargs)
        elif task == "text_generation":
            return self.generate_text(data, **kwargs)
        elif task == "embeddings":
            return self.get_embeddings(data, **kwargs)
        elif task == "function_calling":
            return self.function_calling(data, **kwargs)
        else:
            return f"LLM task '{task}' not supported."
    
    def chat(self, messages, provider="openai", model=None, **kwargs):
        """
        Chat with an LLM in a conversational format
        
        Args:
            messages (list): List of message dictionaries with 'role' and 'content' keys
            provider (str): Provider to use (openai, gemini, huggingface)
            model (str): Specific model to use
            
        Returns:
            dict: Chat response
        """
        if provider == "openai":
            if not self.openai_util:
                return "OpenAI not initialized. Provide a valid API key."
                
            model_name = model or "gpt-3.5-turbo"
            response = self.openai_util.generate_chat_completion(messages, model=model_name, **kwargs)
            return {
                "provider": "openai",
                "model": model_name,
                "response": response,
                "messages": messages
            }
            
        elif provider == "gemini":
            if not self.gemini_util:
                return "Gemini not initialized. Provide a valid API key."
                
            # Convert OpenAI-style messages to Gemini format
            gemini_messages = []
            for msg in messages:
                role = "model" if msg["role"] == "assistant" else msg["role"]
                gemini_messages.append({"role": role, "parts": [{"text": msg["content"]}]})
                
            response = self.gemini_util.chat_completion(gemini_messages, **kwargs)
            return {
                "provider": "gemini",
                "model": "gemini-pro",
                "response": response,
                "messages": messages
            }
            
        elif provider == "huggingface":
            # For HF we need to concatenate the messages into a single prompt
            prompt = self._format_messages_for_hf(messages)
            
            # Get or initialize the chat pipeline
            if not self.hf_chat_pipeline:
                model_name = model or "facebook/blenderbot-400M-distill"
                self.hf_chat_pipeline = get_hf_pipeline(
                    "conversational", 
                    model_name=model_name,
                    token=self.config_manager.get_huggingface_token() if self.config_manager else None
                )
                
            if not self.hf_chat_pipeline:
                return "Failed to initialize Hugging Face conversational pipeline."
                
            response = self.hf_chat_pipeline(prompt)
            
            # Extract the model's reply
            if hasattr(response, "generated_responses"):
                reply = response.generated_responses[-1]
            else:
                reply = str(response)
                
            return {
                "provider": "huggingface",
                "model": model or "facebook/blenderbot-400M-distill",
                "response": reply,
                "messages": messages
            }
            
        else:
            return f"Chat provider '{provider}' not supported."
    
    def question_answering(self, data, provider="huggingface", model=None, **kwargs):
        """
        Answer questions based on provided context
        
        Args:
            data (dict): Dictionary with 'question' and 'context' keys
            provider (str): Provider to use (openai, gemini, huggingface)
            model (str): Specific model to use
            
        Returns:
            dict: Question answering response
        """
        if not isinstance(data, dict) or "question" not in data:
            return "Data should be a dictionary with at least a 'question' key."
            
        question = data["question"]
        context = data.get("context", "")
        
        if provider == "openai":
            if not self.openai_util:
                return "OpenAI not initialized. Provide a valid API key."
                
            model_name = model or "gpt-3.5-turbo"
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            
            messages = [{"role": "user", "content": prompt}]
            response = self.openai_util.generate_chat_completion(messages, model=model_name, **kwargs)
            
            return {
                "provider": "openai",
                "model": model_name,
                "question": question,
                "context": context,
                "answer": response,
                "confidence": None  # OpenAI doesn't provide confidence scores
            }
            
        elif provider == "gemini":
            if not self.gemini_util:
                return "Gemini not initialized. Provide a valid API key."
                
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            response = self.gemini_util.generate_text(prompt, **kwargs)
            
            return {
                "provider": "gemini",
                "model": "gemini-pro",
                "question": question,
                "context": context,
                "answer": response,
                "confidence": None  # Gemini doesn't provide confidence scores
            }
            
        elif provider == "huggingface":
            # Get or initialize the QA pipeline
            if not self.hf_qa_pipeline:
                model_name = model or "deepset/roberta-base-squad2"
                self.hf_qa_pipeline = get_hf_pipeline(
                    "question-answering", 
                    model_name=model_name,
                    token=self.config_manager.get_huggingface_token() if self.config_manager else None
                )
                
            if not self.hf_qa_pipeline:
                return "Failed to initialize Hugging Face question-answering pipeline."
                
            # If no context is provided, return an error
            if not context:
                return "Hugging Face question-answering requires a context."
                
            response = self.hf_qa_pipeline(question=question, context=context)
            
            return {
                "provider": "huggingface",
                "model": model or "deepset/roberta-base-squad2",
                "question": question,
                "context": context,
                "answer": response["answer"],
                "confidence": response["score"]
            }
            
        else:
            return f"Question answering provider '{provider}' not supported."
    
    def summarize(self, text, provider="huggingface", model=None, **kwargs):
        """
        Summarize text
        
        Args:
            text (str): Text to summarize
            provider (str): Provider to use (openai, gemini, huggingface)
            model (str): Specific model to use
            
        Returns:
            dict: Summarization response
        """
        if provider == "openai":
            if not self.openai_util:
                return "OpenAI not initialized. Provide a valid API key."
                
            model_name = model or "gpt-3.5-turbo"
            prompt = f"Summarize the following text:\n\n{text}"
            
            messages = [{"role": "user", "content": prompt}]
            response = self.openai_util.generate_chat_completion(messages, model=model_name, **kwargs)
            
            return {
                "provider": "openai",
                "model": model_name,
                "original_text": text[:1000] + "..." if len(text) > 1000 else text,  # Truncate for clarity
                "summary": response
            }
            
        elif provider == "gemini":
            if not self.gemini_util:
                return "Gemini not initialized. Provide a valid API key."
                
            prompt = f"Summarize the following text:\n\n{text}"
            response = self.gemini_util.generate_text(prompt, **kwargs)
            
            return {
                "provider": "gemini",
                "model": "gemini-pro",
                "original_text": text[:1000] + "..." if len(text) > 1000 else text,
                "summary": response
            }
            
        elif provider == "huggingface":
            # Get or initialize the summarization pipeline
            if not self.hf_summarization_pipeline:
                model_name = model or "facebook/bart-large-cnn"
                self.hf_summarization_pipeline = get_hf_pipeline(
                    "summarization", 
                    model_name=model_name,
                    token=self.config_manager.get_huggingface_token() if self.config_manager else None
                )
                
            if not self.hf_summarization_pipeline:
                return "Failed to initialize Hugging Face summarization pipeline."
                
            # Set summarization parameters
            max_length = kwargs.get("max_length", 130)
            min_length = kwargs.get("min_length", 30)
            
            response = self.hf_summarization_pipeline(
                text, 
                max_length=max_length, 
                min_length=min_length, 
                do_sample=False
            )
            
            # Extract the summary
            if isinstance(response, list):
                summary = response[0]["summary_text"]
            else:
                summary = response["summary_text"]
                
            return {
                "provider": "huggingface",
                "model": model or "facebook/bart-large-cnn",
                "original_text": text[:1000] + "..." if len(text) > 1000 else text,
                "summary": summary
            }
            
        else:
            return f"Summarization provider '{provider}' not supported."
    
    def classify(self, text, labels=None, provider="huggingface", model=None, **kwargs):
        """
        Classify text into categories
        
        Args:
            text (str): Text to classify
            labels (list): List of possible labels (for zero-shot classification)
            provider (str): Provider to use (openai, gemini, huggingface)
            model (str): Specific model to use
            
        Returns:
            dict: Classification response
        """
        if provider == "openai":
            if not self.openai_util:
                return "OpenAI not initialized. Provide a valid API key."
                
            model_name = model or "gpt-3.5-turbo"
            
            if labels:
                prompt = f"Classify the following text into exactly one of these categories: {', '.join(labels)}.\n\nText: {text}\n\nCategory:"
            else:
                prompt = f"Classify the following text into an appropriate category.\n\nText: {text}\n\nCategory:"
                
            messages = [{"role": "user", "content": prompt}]
            response = self.openai_util.generate_chat_completion(messages, model=model_name, **kwargs)
            
            # Try to extract the category from the response
            category = response.strip()
            
            return {
                "provider": "openai",
                "model": model_name,
                "text": text,
                "category": category,
                "confidence": None  # OpenAI doesn't provide confidence scores
            }
            
        elif provider == "gemini":
            if not self.gemini_util:
                return "Gemini not initialized. Provide a valid API key."
                
            if labels:
                prompt = f"Classify the following text into exactly one of these categories: {', '.join(labels)}.\n\nText: {text}\n\nCategory:"
            else:
                prompt = f"Classify the following text into an appropriate category.\n\nText: {text}\n\nCategory:"
                
            response = self.gemini_util.generate_text(prompt, **kwargs)
            
            # Try to extract the category from the response
            category = response.strip()
            
            return {
                "provider": "gemini",
                "model": "gemini-pro",
                "text": text,
                "category": category,
                "confidence": None  # Gemini doesn't provide confidence scores
            }
            
        elif provider == "huggingface":
            # Determine classification type
            if labels:
                # Zero-shot classification
                from transformers import pipeline
                classifier = pipeline(
                    "zero-shot-classification", 
                    model=model or "facebook/bart-large-mnli",
                    token=self.config_manager.get_huggingface_token() if self.config_manager else None
                )
                
                result = classifier(text, labels)
                
                return {
                    "provider": "huggingface",
                    "model": model or "facebook/bart-large-mnli",
                    "text": text,
                    "category": result["labels"][0],
                    "confidence": result["scores"][0],
                    "all_categories": result["labels"],
                    "all_scores": result["scores"]
                }
            else:
                # Standard text classification
                if not self.hf_classification_pipeline:
                    model_name = model or "distilbert-base-uncased-finetuned-sst-2-english"
                    self.hf_classification_pipeline = get_hf_pipeline(
                        "text-classification", 
                        model_name=model_name,
                        token=self.config_manager.get_huggingface_token() if self.config_manager else None
                    )
                    
                if not self.hf_classification_pipeline:
                    return "Failed to initialize Hugging Face classification pipeline."
                    
                result = self.hf_classification_pipeline(text)
                
                if isinstance(result, list):
                    result = result[0]
                    
                return {
                    "provider": "huggingface",
                    "model": model or "distilbert-base-uncased-finetuned-sst-2-english",
                    "text": text,
                    "category": result["label"],
                    "confidence": result["score"]
                }
        else:
            return f"Classification provider '{provider}' not supported."
    
    def generate_text(self, prompt, provider="openai", model=None, **kwargs):
        """
        Generate text based on a prompt
        
        Args:
            prompt (str): Text prompt
            provider (str): Provider to use (openai, gemini, huggingface)
            model (str): Specific model to use
            
        Returns:
            dict: Text generation response
        """
        if provider == "openai":
            if not self.openai_util:
                return "OpenAI not initialized. Provide a valid API key."
                
            model_name = model or "gpt-3.5-turbo"
            messages = [{"role": "user", "content": prompt}]
            response = self.openai_util.generate_chat_completion(messages, model=model_name, **kwargs)
            
            return {
                "provider": "openai",
                "model": model_name,
                "prompt": prompt,
                "generated_text": response
            }
            
        elif provider == "gemini":
            if not self.gemini_util:
                return "Gemini not initialized. Provide a valid API key."
                
            response = self.gemini_util.generate_text(prompt, **kwargs)
            
            return {
                "provider": "gemini",
                "model": "gemini-pro",
                "prompt": prompt,
                "generated_text": response
            }
            
        elif provider == "huggingface":
            # Get or initialize the text generation pipeline
            from transformers import pipeline
            model_name = model or "gpt2"
            generator = pipeline(
                "text-generation", 
                model=model_name,
                token=self.config_manager.get_huggingface_token() if self.config_manager else None
            )
            
            # Set generation parameters
            max_length = kwargs.get("max_length", 100)
            num_return_sequences = kwargs.get("num_return_sequences", 1)
            
            response = generator(
                prompt, 
                max_length=max_length, 
                num_return_sequences=num_return_sequences
            )
            
            generated_texts = [item["generated_text"] for item in response]
            
            return {
                "provider": "huggingface",
                "model": model_name,
                "prompt": prompt,
                "generated_text": generated_texts[0],
                "all_texts": generated_texts if num_return_sequences > 1 else None
            }
            
        else:
            return f"Text generation provider '{provider}' not supported."
    
    def get_embeddings(self, texts, provider="openai", model=None, **kwargs):
        """
        Get vector embeddings for text
        
        Args:
            texts (str or list): Text(s) to embed
            provider (str): Provider to use (openai, huggingface)
            model (str): Specific model to use
            
        Returns:
            dict: Embeddings response
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
            
        if provider == "openai":
            if not self.openai_util:
                return "OpenAI not initialized. Provide a valid API key."
                
            model_name = model or "text-embedding-ada-002"
            embeddings = self.openai_util.create_embeddings(texts, model=model_name)
            
            return {
                "provider": "openai",
                "model": model_name,
                "embeddings": embeddings
            }
            
        elif provider == "huggingface":
            # Use sentence-transformers for embeddings
            try:
                from sentence_transformers import SentenceTransformer
                
                model_name = model or "all-MiniLM-L6-v2"
                embedding_model = SentenceTransformer(model_name)
                
                # Generate embeddings
                embeddings = embedding_model.encode(texts, convert_to_numpy=True)
                
                # Convert to list of lists for JSON serialization
                embeddings_list = embeddings.tolist()
                
                return {
                    "provider": "huggingface",
                    "model": model_name,
                    "embeddings": embeddings_list
                }
                
            except ImportError:
                return "sentence_transformers package not installed. Install with: pip install sentence-transformers"
            except Exception as e:
                return f"Error creating embeddings: {str(e)}"
                
        else:
            return f"Embeddings provider '{provider}' not supported."
    
    def function_calling(self, prompt, functions, provider="openai", model=None, **kwargs):
        """
        Call functions with LLM
        
        Args:
            prompt (str): User prompt
            functions (list): List of function definitions
            provider (str): Provider to use (openai, gemini)
            model (str): Specific model to use
            
        Returns:
            dict: Function calling response
        """
        if provider == "openai":
            if not self.openai_util:
                return "OpenAI not initialized. Provide a valid API key."
                
            model_name = model or "gpt-3.5-turbo"
            messages = [{"role": "user", "content": prompt}]
            
            response = self.openai_util.function_calling(messages, functions, model=model_name, **kwargs)
            
            return {
                "provider": "openai",
                "model": model_name,
                "prompt": prompt,
                "response": response
            }
            
        elif provider == "gemini":
            if not self.gemini_util:
                return "Gemini not initialized. Provide a valid API key."
                
            response = self.gemini_util.function_calling(prompt, functions, **kwargs)
            
            return {
                "provider": "gemini",
                "model": "gemini-pro",
                "prompt": prompt,
                "response": response
            }
            
        else:
            return f"Function calling provider '{provider}' not supported."
    
    def _format_messages_for_hf(self, messages):
        """Format a list of messages for Hugging Face models"""
        formatted = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
                
        return "\n".join(formatted)
