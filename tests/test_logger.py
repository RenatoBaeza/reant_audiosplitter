import os
import logging
import tempfile
import yaml
from utils.logger import setup_logger


def test_logger_creation():
    """Test basic logger creation with default settings."""
    logger = setup_logger("test_logger")
    
    # Check logger properties
    assert logger.name == "test_logger"
    assert logger.level == logging.INFO
    
    # Check handlers
    assert len(logger.handlers) > 0
    
    # At least one handler should be a StreamHandler
    stream_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
    assert len(stream_handlers) > 0
    
    # Clean up handlers
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)


def test_logger_with_config():
    """Test logger creation with custom configuration."""
    # Create a temporary configuration file
    with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w+', delete=False) as temp_file:
        # Write test configuration
        test_config = {
            'logging': {
                'level': 'DEBUG',
                'file': 'test_log.log',
                'console': True
            }
        }
        yaml.dump(test_config, temp_file)
        temp_file_path = temp_file.name
    
    try:
        # Set up logger with the custom config
        logger = setup_logger("test_config_logger", temp_file_path)
        
        # Check logger properties
        assert logger.name == "test_config_logger"
        assert logger.level == logging.DEBUG
        
        # Check handlers
        handlers = logger.handlers
        assert len(handlers) == 2  # One file handler and one stream handler
        
        # Verify file handler
        file_handlers = [h for h in handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1
        assert file_handlers[0].baseFilename.endswith('test_log.log')
        
        # Verify stream handler
        stream_handlers = [h for h in handlers if isinstance(h, logging.StreamHandler) 
                            and not isinstance(h, logging.FileHandler)]
        assert len(stream_handlers) == 1
        
        # Clean up handlers
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)
        
    finally:
        # Clean up
        os.unlink(temp_file_path)
        
        # Remove the log file if it was created
        if os.path.exists('test_log.log'):
            os.unlink('test_log.log')


def test_logger_no_console():
    """Test logger with console output disabled."""
    # Create a temporary configuration file
    with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w+', delete=False) as temp_file:
        # Write test configuration
        test_config = {
            'logging': {
                'level': 'WARNING',
                'file': 'test_log2.log',
                'console': False
            }
        }
        yaml.dump(test_config, temp_file)
        temp_file_path = temp_file.name
    
    try:
        # Set up logger with the custom config
        logger = setup_logger("test_no_console", temp_file_path)
        
        # Check logger properties
        assert logger.name == "test_no_console"
        assert logger.level == logging.WARNING
        
        # Check handlers (should only have file handler)
        handlers = logger.handlers
        assert len(handlers) == 1
        
        # Verify file handler
        file_handlers = [h for h in handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1
        
        # Verify no stream handlers
        stream_handlers = [h for h in handlers if isinstance(h, logging.StreamHandler) 
                            and not isinstance(h, logging.FileHandler)]
        assert len(stream_handlers) == 0
        
        # Clean up handlers
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)
        
    finally:
        # Clean up
        os.unlink(temp_file_path)
        
        # Remove the log file if it was created
        if os.path.exists('test_log2.log'):
            os.unlink('test_log2.log') 