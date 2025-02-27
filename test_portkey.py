"""
Simple test script to verify Portkey integration is working
"""

import yaml
import logging
import traceback
from model_client import ModelClient
from custom_llm import PortkeyLLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_portkey")

def test_model_client():
    """Test direct interaction with ModelClient"""
    try:
        logger.info("=== Testing ModelClient ===")
        
        # Load configuration
        with open('config/portkey.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Create client
        client = ModelClient(config=config)
        
        # Test a simple message
        messages = [{"role": "user", "content": "Hello, this is a test message. Please respond with a short greeting."}]
        logger.info("Sending test message to Portkey...")
        
        response = client.invoke_model("test_agent", messages)
        
        if "error" in response:
            logger.error(f"Test failed with error: {response['error']}")
            return False
        else:
            logger.info(f"Response received: {response['content'][0]['text']}")
            logger.info("ModelClient test successful!")
            return True
            
    except Exception as e:
        logger.error(f"Test failed with exception: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def test_custom_llm():
    """Test the PortkeyLLM integration"""
    try:
        logger.info("=== Testing PortkeyLLM ===")
        
        # Load configuration
        with open('config/portkey.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Create LLM
        llm = PortkeyLLM(config=config)
        
        # Test with a simple prompt
        prompt = "Hello, this is a test message by satish. Please respond with a short greeting."
        logger.info("Sending test prompt to PortkeyLLM...")
        
        response = llm.call(prompt)
        
        if response.startswith("Error:"):
            logger.error(f"Test failed: {response}")
            return False
        else:
            logger.info(f"Response received: {response}")
            logger.info("PortkeyLLM test successful!")
            return True
            
    except Exception as e:
        logger.error(f"Test failed with exception: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logger.info("Starting Portkey integration tests")
    
    # Test ModelClient
    model_client_success = test_model_client()
    
    # Test PortkeyLLM
    custom_llm_success = test_custom_llm()
    
    if model_client_success and custom_llm_success:
        logger.info("All tests passed! Your Portkey integration is working correctly.")
    else:
        logger.error("Some tests failed. Please check the errors above.")