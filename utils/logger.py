import logging
import os
import yaml
from pathlib import Path

def setup_logger(name="audiosplitter", config_file="config.yaml"):
    """
    Set up and configure logger based on the configuration file.
    
    Args:
        name (str): Logger name
        config_file (str): Path to the YAML configuration file
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # Default configuration
    log_level = logging.INFO
    log_file = "audiosplitter.log"
    console_logging = True
    
    # Load configuration if available
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            if config and 'logging' in config:
                log_config = config['logging']
                
                # Get log level
                level_str = log_config.get('level', 'INFO')
                log_level = getattr(logging, level_str)
                
                # Get log file
                log_file = log_config.get('file', 'audiosplitter.log')
                
                # Get console logging flag
                console_logging = log_config.get('console', True)
        except Exception as e:
            print(f"Warning: Could not load configuration from {config_file}: {e}")
    
    # Set logger level
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates when called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler
    if console_logging:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger 