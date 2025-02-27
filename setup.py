"""
Setup script for the CrewAI with Portkey integration
"""

import os
import sys
import subprocess

def ensure_directory(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def main():
    # Ensure required directories exist
    ensure_directory("config")
    
    # Install requirements
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    print("Setup complete!")
    print("\nDirectory structure:")
    print("- config/")
    print("  - agents.yaml")
    print("  - tasks.yaml")
    print("  - portkey.yaml")
    print("- main.py")
    print("- crew.py")
    print("- custom_llm.py")
    print("- model_client.py")
    print("- PDFReaderTool.py")
    print("\nTo run the application:")
    print("python main.py")
    print("\nTo train the crew:")
    print("python main.py train 10")

if __name__ == "__main__":
    main()