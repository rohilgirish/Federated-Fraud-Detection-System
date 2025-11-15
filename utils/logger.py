"""
Logging configuration for FairFinance
Provides centralized logging setup for all components
"""

import logging
import os
from datetime import datetime

def setup_logging(name, log_dir='logs'):
    """
    Setup logging for a module
    
    Args:
        name: Module name (e.g., 'server', 'client')
        log_dir: Directory to store log files
    
    Returns:
        logger object
    """
    
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Create file handler (logs to file)
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    # Create console handler (logs to console)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers to logger
    if not logger.handlers:  # Avoid duplicate handlers
        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger


# Example usage
if __name__ == "__main__":
    logger = setup_logging('test')
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
