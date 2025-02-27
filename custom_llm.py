"""
Custom LLM Provider for CrewAI using Portkey - Fixed for LiteLLM usage
"""

import inspect
import os
from typing import Dict, List, Any, Optional, Union

# Import CrewAI LLM class
from crewai import LLM

from model_client import ModelClient

class PortkeyLLM(LLM):
    """
    Custom LLM provider for CrewAI using Portkey.
    Fixed to work with LiteLLM backend in CrewAI.
    """
    
    def __init__(
        self, 
        model: str = "gpt-4o", 
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the PortkeyLLM.
        
        Args:
            model: Model identifier (e.g., "gpt-4o")
            config: Configuration dictionary for ModelClient
            config_path: Path to configuration file for ModelClient
            **kwargs: Additional arguments
        """
        # Use provider "openai" which is handled by LiteLLM
        kwargs["provider"] = "custom"
        
        # Initialize with parent class using custom provider
        super().__init__(model=model, **kwargs)
        
        # Store model configuration
        self.model_name = model
        self.config = config or {}
        
        # Initialize ModelClient for direct API access
        self.model_client = ModelClient(config=config, config_path=config_path)
        
        # Set environment variables that LiteLLM might be looking for
        if config and "portkey" in config and "api" in config["portkey"]:
            api_config = config["portkey"]["api"]
            os.environ["OPENAI_API_KEY"] = api_config.get("api_key", "")
            
            # Set Portkey-specific environment variables
            os.environ["PORTKEY_API_KEY"] = api_config.get("api_key", "")
            os.environ["PORTKEY_BASE_URL"] = api_config.get("base_url", "https://api.portkey.ai/v1")
            os.environ["PORTKEY_VIRTUAL_KEY"] = api_config.get("virtual_key", "")
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate a response using direct Portkey API call instead of LiteLLM.
        
        Args:
            messages: List of messages in the conversation
            **kwargs: Additional arguments
            
        Returns:
            Generated text response
        """
        try:
            # Use the agent_name as the current role or default to "agent"
            agent_name = kwargs.get("agent_name", "agent")
            
            # Directly invoke the model through our ModelClient
            response = self.model_client.invoke_model(agent_name, messages)
            
            # Extract the text content from the response
            if "content" in response and len(response["content"]) > 0:
                return response["content"][0]["text"]
            else:
                return "No response generated"
                
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    # Override call method to bypass LiteLLM completely
    def call(self, prompt: Union[str, List[Dict[str, str]]], **kwargs) -> str:
        """Bypass LiteLLM and use direct Portkey API calls"""
        messages = []
        
        # If prompt is a string, convert to a list with a single user message
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
            
        return self.chat(messages, **kwargs)
    
    def get_model_name(self) -> str:
        """Get the name of the model being used."""
        return self.model_name