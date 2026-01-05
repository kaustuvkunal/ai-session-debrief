"""
LLM Client Wrapper

Handles interactions with OpenAI API using the 'responses' API pattern.
Specific support for o3-mini with reasoning steps.
"""

import os
import logging
from typing import Optional, Type, TypeVar, Any, Dict, List
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import src.config as config

load_dotenv()
logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

def get_client() -> OpenAI:
    """
    Get OpenAI client instance.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not found in environment variables.")
    
    return OpenAI(api_key=api_key)


def generate_structured_response(
    system_prompt: str,
    user_prompt: str,
    response_model: Type[T],
    model_name: Optional[str] = None
) -> T:
    """
    Generate a structured response using OpenAI's client.responses.parse.
    
    Args:
        system_prompt: Instructions for the model
        user_prompt: Content to process
        response_model: Pydantic model class for output validation
        model_name: Override default model from config
        
    Returns:
        Instance of response_model
    """
    client: OpenAI = get_client()
    model: str = model_name or config.MODEL_NAME
    
    logger.debug(f"Generating structured response with model: {model}")
    
    try:
        response = client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text_format=response_model,
           
        )
        
        parsed_response = response.output_parsed
        if not parsed_response:
            raise ValueError("Model returned empty parsed response")
            
        return parsed_response
        
    except Exception as e:
        logger.error(f"Error in structured generation: {e}")
        raise


def generate_text_response(
    system_prompt: str,
    user_prompt: str,
    model_name: Optional[str] = None
) -> str:
    """
    Generate a text response using client.responses.create.
    
    Args:
        system_prompt: Instructions for the model
        user_prompt: Content to process
        model_name: Override default model from config
        
    Returns:
        Generated text string
    """
    client: OpenAI = get_client()
    model: str = model_name or config.MODEL_NAME
    
    logger.debug(f"Generating text response with model: {model}")
    
    try:
        
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        
        # Checking for output attribute vs output_text  
        if hasattr(response, 'output_text') and response.output_text:
            return response.output_text
        elif hasattr(response, 'output') and response.output:
             return response.output
        else:
            raise ValueError("Model returned empty content")
            
    except Exception as e:
        logger.error(f"Error in text generation: {e}")
        raise
