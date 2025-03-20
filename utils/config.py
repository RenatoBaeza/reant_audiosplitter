import os
import yaml
from pathlib import Path

class Config:
    """Configuration manager for audio splitter application."""
    
    def __init__(self, config_file="config.yaml"):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_file (str): Path to the YAML configuration file
        """
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            raise ValueError(f"Failed to load configuration: {e}")
    
    def get_model_config(self):
        """Get model configuration."""
        return self.config.get('model', {})
    
    def get_processing_config(self):
        """Get processing configuration."""
        return self.config.get('processing', {})
    
    def get_output_config(self):
        """Get output configuration."""
        return self.config.get('output', {})
    
    def get_logging_config(self):
        """Get logging configuration."""
        return self.config.get('logging', {})
    
    def get(self, key, default=None):
        """
        Get a configuration value by key with optional default.
        
        Args:
            key (str): Configuration key (can use dot notation for nested keys)
            default: Default value if key not found
            
        Returns:
            Value for the key or default if not found
        """
        if '.' not in key:
            return self.config.get(key, default)
        
        # Handle nested keys with dot notation
        parts = key.split('.')
        value = self.config
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        
        return value 