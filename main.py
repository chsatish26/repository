"""
Main entry point for the Risk Assessment Crew with Portkey integration
"""

import sys
import logging
import traceback
from crew import RiskAssesCrew

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("main")

def run():
    """
    Run the Risk Assessment Crew with Portkey integration.
    """
    try:
        logger.info("Starting Risk Assessment with Portkey integration")
        
        # Initialize empty inputs - task-specific data will be handled by agents
        inputs = {}
        
        # Create and run the crew
        crew_instance = RiskAssesCrew().crew()
        result = crew_instance.kickoff(inputs=inputs)
        
        logger.info("Risk Assessment completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"An error occurred while running the crew: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

def train(iterations=10):
    """
    Train the crew for a given number of iterations.
    The actual training data and parameters are handled by the agents.
    
    Args:
        iterations: Number of training iterations to run
    """
    try:
        logger.info(f"Starting training with {iterations} iterations")
        
        # Empty inputs as configuration is handled in agents and tasks
        inputs = {}
        
        # Create and train the crew
        crew_instance = RiskAssesCrew().crew()
        result = crew_instance.train(n_iterations=iterations, inputs=inputs)
        
        logger.info("Training completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"An error occurred while training the crew: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    # If you wish to train instead of run, call: python main.py train 10
    # (Change '10' to however many iterations you want to train.)
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        train(iterations)
    elif len(sys.argv) > 1 and sys.argv[1] == "troubleshoot":
        # Run the troubleshooting script if requested
        from troubleshoot import main as troubleshoot_main
        troubleshoot_main()
    else:
        run()