import os
import pytest
import tempfile
import yaml
from utils.config import Config


def test_config_loading():
    """Test loading configuration from a file."""
    # Create a temporary configuration file
    with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w+', delete=False) as temp_file:
        # Write test configuration
        test_config = {
            'model': {
                'name': 'test_model',
                'device': 'cpu',
                'shifts': 1
            },
            'processing': {
                'chunk_size': 4
            },
            'output': {
                'directory': 'test_output'
            }
        }
        yaml.dump(test_config, temp_file)
        temp_file_path = temp_file.name
    
    try:
        # Load configuration
        config = Config(temp_file_path)
        
        # Test getting sections
        assert config.get_model_config().get('name') == 'test_model'
        assert config.get_model_config().get('device') == 'cpu'
        assert config.get_model_config().get('shifts') == 1
        
        assert config.get_processing_config().get('chunk_size') == 4
        
        assert config.get_output_config().get('directory') == 'test_output'
        
        # Test get method with dot notation
        assert config.get('model.name') == 'test_model'
        assert config.get('model.device') == 'cpu'
        assert config.get('model.nonexistent', 'default') == 'default'
        
    finally:
        # Clean up
        os.unlink(temp_file_path)


def test_config_file_not_found():
    """Test handling of non-existent configuration file."""
    with pytest.raises(FileNotFoundError):
        Config('nonexistent_file.yaml')


def test_config_invalid_yaml():
    """Test handling of invalid YAML in configuration file."""
    # Create a temporary file with invalid YAML
    with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w+', delete=False) as temp_file:
        temp_file.write('invalid: yaml: content:')
        temp_file_path = temp_file.name
    
    try:
        with pytest.raises(ValueError):
            Config(temp_file_path)
    finally:
        # Clean up
        os.unlink(temp_file_path) 