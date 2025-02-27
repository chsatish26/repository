"""
Model Client for Portkey-OpenAI Integration - Fixed version
"""

import yaml
import json
import logging
import traceback
from typing import Dict, Any, Optional, List, Union

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("model_client")

# Import Portkey inside a try/except to handle import errors gracefully
try:
    from portkey_ai import Portkey
except ImportError:
    logger.error("Failed to import Portkey. Make sure it's installed: pip install portkey-ai")
    Portkey = None

class ModelClient:
    """Unified client for accessing LLM models via Portkey."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None):
        """
        Initialize the model client.
        
        Args:
            config: Configuration dictionary (optional)
            config_path: Path to configuration file (optional)
        """
        logger.info("Initializing ModelClient")
        
        try:
            if config:
                self.config = config
                logger.info("Using provided configuration dictionary")
            elif config_path:
                with open(config_path, "r") as f:
                    self.config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_path}")
            else:
                self.config = {}
                logger.warning("No configuration provided, using empty config")
            
            self.portkey_client = None  # Initialize Portkey client
            self.model_errors = []
            self._init_clients()
            
        except Exception as e:
            logger.error(f"Error initializing ModelClient: {str(e)}")
            logger.error(traceback.format_exc())
            self.model_errors.append(f"Initialization error: {str(e)}")
            # Don't raise the exception here, allow the class to be instantiated
    
    def _init_clients(self):
        """Initialize necessary clients based on configuration."""
        if Portkey is None:
            logger.error("Cannot initialize Portkey: Module not found")
            return
            
        logger.info("Initializing Portkey client...")
        
        # Initialize Portkey - enable by default if not specified
        portkey_enabled = self.config.get('portkey', {}).get('enabled', True)
        
        if portkey_enabled:
            try:
                portkey_config = self.config.get('portkey', {}).get('api', {})
                
                base_url = portkey_config.get('base_url', "https://api.portkey.ai/v1")
                api_key = portkey_config.get('api_key', "cmG+")
                virtual_key = portkey_config.get('virtual_key', "open-ai-virtua")
                
                logger.info(f"Initializing Portkey with base_url: {base_url}")
                logger.info(f"Using virtual_key: {virtual_key}")
                
                # Create a direct Portkey client
                self.portkey_client = Portkey(
                    base_url=base_url,
                    api_key=api_key,
                    virtual_key=virtual_key
                )
                
                logger.info("Portkey client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Portkey client: {e}")
                logger.error(traceback.format_exc())
                self.model_errors.append(f"Portkey initialization error: {str(e)}")
                self.portkey_client = None
        else:
            logger.info("Portkey is not enabled in configuration")
    
    def _format_messages(self, messages: List[Union[Dict[str, Any], str]]) -> List[Dict[str, str]]:
        """
        Format messages to ensure compatibility with Portkey API.
        
        Args:
            messages: List of messages that might be in different formats
            
        Returns:
            List of properly formatted messages for Portkey
        """
        formatted_messages = []
        
        try:
            for msg in messages:
                # Handle different message formats
                if isinstance(msg, dict):
                    # CrewAI might send different formats
                    if 'role' in msg and 'content' in msg:
                        # Standard format {role: ..., content: ...}
                        formatted_msg = {
                            'role': msg['role'],
                            'content': msg['content']
                        }
                    elif 'type' in msg and 'data' in msg:
                        # Alternative format {type: ..., data: {...}}
                        if isinstance(msg['data'], dict) and 'content' in msg['data']:
                            formatted_msg = {
                                'role': msg.get('type', 'user'),
                                'content': msg['data']['content']
                            }
                        else:
                            formatted_msg = {
                                'role': msg.get('type', 'user'),
                                'content': str(msg['data'])
                            }
                    else:
                        # Unknown format, use as is with defaults
                        formatted_msg = {
                            'role': msg.get('role', 'user'),
                            'content': str(msg.get('content', str(msg)))
                        }
                else:
                    # If not a dict, create a user message with the string representation
                    formatted_msg = {
                        'role': 'user',
                        'content': str(msg)
                    }
                
                # Ensure content is a string
                if not isinstance(formatted_msg['content'], str):
                    formatted_msg['content'] = str(formatted_msg['content'])
                
                formatted_messages.append(formatted_msg)
                
            return formatted_messages
        except Exception as e:
            logger.error(f"Error formatting messages: {str(e)}")
            logger.error(traceback.format_exc())
            # Return default message if formatting fails
            return [{'role': 'user', 'content': 'Error formatting messages'}]
    
    def invoke_model(self, agent_name: str, messages: List[Union[Dict[str, Any], str]]) -> Dict[str, Any]:
        """
        Invoke the model using Portkey.
        
        Args:
            agent_name: Name of the agent (for logging purposes)
            messages: List of messages to send to the model
            
        Returns:
            Dictionary containing the model's response
        """
        try:
            if not self.portkey_client:
                raise ValueError("Portkey client is not initialized")
            
            # Format messages to ensure compatibility
            formatted_messages = self._format_messages(messages)
            
            # Get model configuration
            model_config = self.config.get('portkey', {}).get('model', {})
            model_name = model_config.get('name', 'gpt-4o')
            max_tokens = model_config.get('max_tokens', 3000)
            temperature = model_config.get('temperature', 0)
            
            logger.info(f"Invoking model via Portkey for agent '{agent_name}'")
            logger.info(f"Model: {model_name}, Max tokens: {max_tokens}, Temperature: {temperature}")
            
            # Make direct API call to avoid LiteLLM
            response = self.portkey_client.chat.completions.create(
                messages=formatted_messages,
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            logger.info(f"Received response from Portkey")
            
            # Extract response content
            response_content = response.choices[0].message.content
            
            return {
                "content": [{"text": response_content}],
                "model_used": "portkey"
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Portkey invocation failed: {error_msg}")
            logger.error(traceback.format_exc())
            
            return {
                "content": [{"text": f"Error: {error_msg}"}],
                "error": error_msg,
                "model_used": "error"
            }
    
    def get_errors(self) -> list:
        """Get list of errors encountered."""
        return self.model_errors