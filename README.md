# CrewAI with Portkey Integration

This project integrates CrewAI with Portkey to provide access to various LLM providers through a unified interface.

## Overview

This implementation combines the power of CrewAI's agent-based workflow with Portkey's model management capabilities. The main components are:

1. **Custom LLM Provider**: Integrates with Portkey to access various models
2. **ModelClient**: Handles communication with the Portkey API
3. **Risk Assessment Crew**: A sample CrewAI application that performs credit risk assessment

## Setup Instructions

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone the repository and navigate to the project directory

2. Install required packages:
   ```bash
   pip install crewai portkey-ai PyPDF2 pyyaml
   ```

3. Configure Portkey API credentials:
   - Edit `config/portkey.yaml` and update your API keys:
     ```yaml
     portkey:
       enabled: true
       api:
         base_url: "https://api.portkey.ai/v1"
         api_key: "YOUR_API_KEY"
         virtual_key: "YOUR_VIRTUAL_KEY"
       model:
         name: "gpt-4o"  # Or your preferred model
         max_tokens: 3000
         temperature: 0
     ```

4. Test your Portkey integration:
   ```bash
   python test_portkey.py
   ```

### Running the Application

To run the risk assessment application:
```bash
python main.py
```

To train the crew (if supported by your agents):
```bash
python main.py train 10
```

## File Structure

- `main.py` - Entry point for the application
- `crew.py` - Contains the RiskAssesCrew class and related setup
- `model_client.py` - Handles communication with Portkey API
- `custom_llm.py` - Custom LLM provider for CrewAI
- `PDFReaderTool.py` - Utility for reading PDF files
- `test_portkey.py` - Test script for verifying Portkey integration
- `config/` - Configuration directory
  - `portkey.yaml` - Portkey API configuration
  - `agents.yaml` - Agent configurations for CrewAI
  - `tasks.yaml` - Task configurations for CrewAI

## Configuration

### Portkey Configuration

The `config/portkey.yaml` file controls the Portkey integration:

```yaml
portkey:
  enabled: true  # Set to false to disable Portkey
  
  api:
    base_url: "https://api.portkey.ai/v1"
    api_key: "YOUR_API_KEY"
    virtual_key: "YOUR_VIRTUAL_KEY"
  
  model:
    name: "gpt-4o"  # Model to use
    max_tokens: 3000  # Response length limit
    temperature: 0  # Randomness (0 = deterministic)
```

### Agent Configuration

The `config/agents.yaml` file defines agent capabilities:

```yaml
risk_agent:
  role: "Senior Credit Analyst"
  goal: "Conduct a comprehensive analysis of a provided credit report"
  backstory: "Expert in analyzing credit reports"

# Additional agents...
```

### Task Configuration

The `config/tasks.yaml` file defines task requirements:

```yaml
research_credit_report_task:
  description: "Analyze credit report to identify key factors"
  expected_output: "A concise list of key factors"

# Additional tasks...
```

## Troubleshooting

If you encounter issues, first run the test script:

```bash
python test_portkey.py
```

Common issues and solutions:

1. **Authentication Errors**: 
   - Verify your Portkey API key and virtual key in the config file
   - Make sure your Portkey account has access to the model you're trying to use

2. **Missing Dependencies**: 
   - Run `pip install crewai portkey-ai PyPDF2 pyyaml`

3. **Configuration Errors**: 
   - Check your YAML files for syntax errors
   - Make sure the directory structure is correct

4. **LiteLLM Errors**: 
   - The custom implementation bypasses LiteLLM to avoid authentication issues
   - If you see LiteLLM errors, make sure you're using the latest version of `custom_llm.py`

## Customization

### Using Different Models

To use a different model, update the `name` field in the `model` section of `config/portkey.yaml`:

```yaml
model:
  name: "gpt-4-turbo"  # Change to your preferred model
  max_tokens: 3000
  temperature: 0
```

### Adding New Agents or Tasks

To add new agents or tasks, update the respective files:
- Add agent configurations to `config/agents.yaml`
- Add task configurations to `config/tasks.yaml`
- Update the `crew.py` file to include the new agents and tasks