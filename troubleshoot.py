"""
Troubleshooting script for CrewAI Portkey integration
"""

import os
import sys
import yaml
import logging
import importlib
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("troubleshoot")

def check_imports():
    """Check if all required packages are installed"""
    required_packages = ["crewai", "portkey_ai", "PyPDF2", "pyyaml"]
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            logger.info(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package} is not installed")
    
    return missing_packages

def check_config_files():
    """Check if configuration files exist and are valid"""
    config_files = [
        "config/portkey.yaml",
        "config/agents.yaml",
        "config/tasks.yaml"
    ]
    
    missing_files = []
    
    for config_file in config_files:
        if not os.path.exists(config_file):
            missing_files.append(config_file)
            logger.error(f"❌ Configuration file {config_file} not found")
        else:
            try:
                with open(config_file, 'r') as f:
                    yaml.safe_load(f)
                logger.info(f"✅ Configuration file {config_file} exists and is valid YAML")
            except yaml.YAMLError:
                logger.error(f"❌ Configuration file {config_file} is not valid YAML")
                missing_files.append(f"{config_file} (invalid YAML)")
    
    return missing_files

def check_portkey_connection():
    """Check if Portkey connection works"""
    try:
        from portkey_ai import Portkey
        
        # Load Portkey configuration
        try:
            with open('config/portkey.yaml', 'r') as f:
                config = yaml.safe_load(f)
                
            portkey_config = config.get('portkey', {}).get('api', {})
            base_url = portkey_config.get('base_url', "https://api.portkey.ai/v1")
            api_key = portkey_config.get('api_key', "")
            virtual_key = portkey_config.get('virtual_key', "")
            
            logger.info(f"Testing Portkey connection with base_url: {base_url}")
            logger.info(f"Using virtual_key: {virtual_key}")
            
            client = Portkey(
                base_url=base_url,
                api_key=api_key,
                virtual_key=virtual_key
            )
            
            # Simple test request
            messages = [{"role": "user", "content": "Hello, this is a test."}]
            response = client.chat.completions.create(
                messages=messages,
                model="gpt-4o",
                max_tokens=10,
                temperature=0
            )
            
            logger.info(f"✅ Portkey connection successful")
            logger.info(f"Response: {response.choices[0].message.content}")
            return True
        
        except Exception as e:
            logger.error(f"❌ Portkey connection failed: {str(e)}")
            logger.error(traceback.format_exc())
            return False
            
    except ImportError:
        logger.error("❌ portkey_ai module not installed")
        return False

def check_model_client():
    """Test the ModelClient class"""
    try:
        from model_client import ModelClient
        
        logger.info("Creating ModelClient instance...")
        client = ModelClient()
        
        if client.portkey_client:
            logger.info("✅ ModelClient initialized successfully")
            
            # Test invoking the model
            messages = [{"role": "user", "content": "Hello, this is a test."}]
            response = client.invoke_model("test_agent", messages)
            
            if "error" in response:
                logger.error(f"❌ ModelClient invoke_model failed: {response['error']}")
                return False
            else:
                logger.info(f"✅ ModelClient invoke_model successful")
                logger.info(f"Response: {response['content'][0]['text']}")
                return True
        else:
            logger.error("❌ ModelClient failed to initialize Portkey client")
            return False
            
    except Exception as e:
        logger.error(f"❌ ModelClient test failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def check_custom_llm():
    """Test the custom LLM class"""
    try:
        from custom_llm import PortkeyLLM
        
        logger.info("Creating PortkeyLLM instance...")
        llm = PortkeyLLM()
        
        logger.info(f"✅ PortkeyLLM initialized successfully")
        logger.info(f"Model name: {llm.get_model_name()}")
        
        # Check available methods
        methods = [name for name in dir(llm) if not name.startswith('_') and callable(getattr(llm, name))]
        logger.info(f"Available methods: {methods}")
        
        # Test the LLM
        if hasattr(llm, 'chat'):
            messages = [{"role": "user", "content": "Hello, this is a test."}]
            response = llm.chat(messages)
            logger.info(f"✅ PortkeyLLM chat method successful")
            logger.info(f"Response: {response}")
            return True
        elif hasattr(llm, 'generate'):
            messages = [{"role": "user", "content": "Hello, this is a test."}]
            response = llm.generate(messages)
            logger.info(f"✅ PortkeyLLM generate method successful")
            logger.info(f"Response: {response}")
            return True
        else:
            logger.error("❌ PortkeyLLM has neither chat nor generate method")
            return False
            
    except Exception as e:
        logger.error(f"❌ PortkeyLLM test failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def fix_missing_packages(missing_packages):
    """Install missing packages"""
    if missing_packages:
        logger.info("Installing missing packages...")
        
        for package in missing_packages:
            try:
                logger.info(f"Installing {package}...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                logger.info(f"✅ Installed {package}")
            except Exception as e:
                logger.error(f"❌ Failed to install {package}: {str(e)}")

def fix_missing_config_files(missing_files):
    """Create missing configuration files with default values"""
    for file in missing_files:
        if file.endswith(" (invalid YAML)"):
            file = file.replace(" (invalid YAML)", "")
            logger.info(f"Fixing invalid YAML in {file}...")
        else:
            logger.info(f"Creating missing config file {file}...")
        
        os.makedirs(os.path.dirname(file), exist_ok=True)
        
        if file == "config/portkey.yaml":
            config = {
                "portkey": {
                    "enabled": True,
                    "api": {
                        "base_url": "https://api.portkey.ai/v1",
                        "api_key": "cmG+Qao3xQygGAZhnEJNN9WaOj6I",
                        "virtual_key": "open-ai-virtual-350afa"
                    },
                    "model": {
                        "name": "gpt-4o",
                        "max_tokens": 3000,
                        "temperature": 0
                    }
                }
            }
        elif file == "config/agents.yaml" and not os.path.exists(file):
            # Only create if it doesn't exist, as it might have been supplied by the user
            config = {
                "risk_agent": {
                    "role": "Senior Credit Analyst",
                    "goal": "Conduct a comprehensive analysis of a provided credit report",
                    "backstory": "Expert in analyzing credit reports"
                },
                "loan_agent": {
                    "role": "Senior Loan Officer",
                    "goal": "Write a concise credit decision report",
                    "backstory": "Skilled in crafting loan applicant's decision report"
                }
            }
        elif file == "config/tasks.yaml" and not os.path.exists(file):
            # Only create if it doesn't exist, as it might have been supplied by the user
            config = {
                "research_credit_report_task": {
                    "description": "Analyze credit report to identify key factors",
                    "expected_output": "A concise list of key factors"
                },
                "loan_assesment_task": {
                    "description": "Write a credit decision report",
                    "expected_output": "A concise credit decision report"
                }
            }
        else:
            # We already have the file or it's not one we want to create
            continue
        
        with open(file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        logger.info(f"✅ Created/fixed {file}")

def main():
    """Main troubleshooting function"""
    logger.info("Starting troubleshooting...")
    
    # Step 1: Check imports
    logger.info("\n=== Checking required packages ===")
    missing_packages = check_imports()
    
    if missing_packages:
        logger.info("\n=== Installing missing packages ===")
        fix_missing_packages(missing_packages)
    
    # Step 2: Check config files
    logger.info("\n=== Checking configuration files ===")
    missing_files = check_config_files()
    
    if missing_files:
        logger.info("\n=== Creating missing configuration files ===")
        fix_missing_config_files(missing_files)
    
    # Step 3: Check Portkey connection
    logger.info("\n=== Testing Portkey connection ===")
    portkey_works = check_portkey_connection()
    
    if not portkey_works:
        logger.error("Portkey connection failed. Please check your API keys and configuration.")
        return
    
    # Step 4: Test ModelClient
    logger.info("\n=== Testing ModelClient ===")
    model_client_works = check_model_client()
    
    if not model_client_works:
        logger.error("ModelClient test failed. Please check the errors above.")
        return
    
    # Step 5: Test Custom LLM
    logger.info("\n=== Testing Custom LLM ===")
    custom_llm_works = check_custom_llm()
    
    if not custom_llm_works:
        logger.error("Custom LLM test failed. Please check the errors above.")
        return
    
    logger.info("\n=== All tests passed! ===")
    logger.info("Your CrewAI Portkey integration should be working correctly.")
    logger.info("You can now run `python main.py` to start your application.")

if __name__ == "__main__":
    main()