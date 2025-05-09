from fastapi import APIRouter, Depends, HTTPException
import time
from typing import Any, Dict, List

from hackaikit.api.schemas import (
    ChatRequest,
    ChatResponse,
    GenerateRequest,
    TextOutputResponse
)
from hackaikit.core.module_manager import ModuleManager

# Create router
router = APIRouter()

# Dependency to get the module_manager from main app
def get_module_manager():
    from hackaikit.api.main import app
    return app.state.module_manager

# Dependency to get the LLMToolsModule instance
def get_llm_module(module_manager: ModuleManager = Depends(get_module_manager)):
    try:
        return module_manager.get_module("LLMToolsModule")
    except ValueError as e:
        raise HTTPException(status_code=404, detail="LLM module not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading LLM module: {str(e)}")

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    llm_module = Depends(get_llm_module)
):
    """
    Interact with an LLM in a conversational format.
    
    Messages should include role (system, user, assistant) and content.
    Provider can be openai, gemini, or huggingface.
    """
    try:
        start_time = time.time()
        
        # Convert Pydantic model to dict for module processing
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Prepare parameters
        params = {}
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.max_tokens is not None:
            params["max_tokens"] = request.max_tokens
            
        # Process with the LLM module
        result = llm_module.process(
            data=messages,
            task="chat",
            provider=request.provider.value,
            model=request.model,
            **params
        )
        
        # Check for error (string response indicates error)
        if isinstance(result, str):
            raise HTTPException(status_code=400, detail=result)
            
        # Extract response from result based on provider
        if request.provider.value == "openai":
            response_text = result.get("response", "")
            model = result.get("model", request.model or "gpt-3.5-turbo")
        elif request.provider.value == "gemini":
            response_text = result.get("response", "")
            model = result.get("model", request.model or "gemini-pro")
        else:  # huggingface
            response_text = result.get("response", "")
            model = result.get("model", request.model or "default-model")
        
        return ChatResponse(
            response=response_text,
            provider=request.provider,
            model=model
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

@router.post("/generate", response_model=TextOutputResponse)
async def generate_text(
    request: GenerateRequest,
    llm_module = Depends(get_llm_module)
):
    """
    Generate text using an LLM given a prompt.
    
    Provider can be openai, gemini, or huggingface.
    """
    try:
        # Prepare parameters
        params = request.params or {}
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.max_tokens is not None:
            params["max_tokens"] = request.max_tokens
            
        # Process with the LLM module
        result = llm_module.process(
            data=request.prompt,
            task="text_generation",
            provider=request.provider.value,
            model=request.model,
            **params
        )
        
        # Check for error
        if isinstance(result, str):
            raise HTTPException(status_code=400, detail=result)
            
        # Extract generated text from result
        if "generated_text" in result:
            text = result["generated_text"]
        else:
            text = str(result)
            
        # Get model information
        model = result.get("model", request.model)
        
        return TextOutputResponse(
            text=text,
            provider=request.provider,
            model=model
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")

@router.post("/summarize", response_model=TextOutputResponse)
async def summarize_text(
    request: TextOutputResponse,
    llm_module = Depends(get_llm_module)
):
    """
    Summarize text using an LLM.
    
    Provider can be openai, gemini, or huggingface.
    """
    try:
        # Process with the LLM module
        result = llm_module.process(
            data=request.text,
            task="summarization",
            provider=request.provider.value,
            model=request.model
        )
        
        # Check for error
        if isinstance(result, str):
            raise HTTPException(status_code=400, detail=result)
            
        # Extract summary from result
        if "summary" in result:
            summary = result["summary"]
        else:
            summary = str(result)
            
        # Get model information
        model = result.get("model", request.model)
        
        return TextOutputResponse(
            text=summary,
            provider=request.provider,
            model=model
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error summarizing text: {str(e)}") 