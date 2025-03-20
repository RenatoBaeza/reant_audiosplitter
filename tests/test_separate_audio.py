import os
import pytest
import torch
import tempfile
from unittest.mock import patch, MagicMock
import separate_audio


def test_parse_arguments():
    """Test argument parsing."""
    with patch('argparse.ArgumentParser.parse_args') as mock_parse_args:
        # Setup the mock return value
        mock_args = MagicMock()
        mock_args.input = "test.mp3"
        mock_args.model = "htdemucs"
        mock_args.device = "cpu"
        mock_args.config = "config.yaml"
        mock_parse_args.return_value = mock_args
        
        # Call the function
        args = separate_audio.parse_arguments()
        
        # Check the results
        assert args.input == "test.mp3"
        assert args.model == "htdemucs"
        assert args.device == "cpu"
        assert args.config == "config.yaml"


def test_load_config_with_args():
    """Test configuration loading with command line arguments."""
    # Create a temporary configuration file
    with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w+', delete=False) as temp_file:
        temp_file.write("""
model:
  name: htdemucs
  device: cpu
  shifts: 2
  split: false
  overlap: 0.25
processing:
  chunk_size: 6
output:
  directory: separated
        """)
        temp_file_path = temp_file.name
    
    try:
        # Create mock args
        mock_args = MagicMock()
        mock_args.config = temp_file_path
        mock_args.input = "custom.mp3"
        mock_args.model = None
        mock_args.device = "cuda"
        mock_args.shifts = None
        mock_args.split = False
        mock_args.overlap = 0.5
        mock_args.chunk_size = None
        mock_args.output_dir = "custom_output"
        
        # Call the function
        config = separate_audio.load_config_with_args(mock_args)
        
        # Check results
        assert config['input_file'] == "custom.mp3"
        assert config['model']['name'] == "htdemucs"
        assert config['model']['device'] == "cuda"  # Should be overridden
        assert config['model']['overlap'] == 0.5    # Should be overridden
        assert config['output']['directory'] == "custom_output"  # Should be overridden
        
    finally:
        # Clean up
        os.unlink(temp_file_path)


@patch('torch.cuda.is_available', return_value=False)
def test_process_chunk(mock_cuda):
    """Test chunk processing."""
    # Create a mock model
    mock_model = MagicMock()
    
    # Mock apply_model function
    with patch('separate_audio.apply_model') as mock_apply_model:
        # Set up mock return value (2 sources, batch of 1, 2 channels, 1000 samples)
        mock_sources = torch.zeros((2, 2, 1000))
        mock_sources[0, :, :] = 0.5  # Set first source to a constant
        mock_sources[1, :, :] = 0.2  # Set second source to a different constant
        mock_apply_model.return_value = mock_sources
        
        # Create a test chunk (2 channels, 1000 samples)
        test_chunk = torch.zeros((2, 1000))
        
        # Process the chunk
        result = separate_audio.process_chunk(
            test_chunk, 
            mock_model, 
            "cpu", 
            2,  # shifts
            False,  # split
            0.25  # overlap
        )
        
        # Check the result
        assert result.shape == (2, 2, 1000)  # 2 sources, 2 channels, 1000 samples
        assert torch.allclose(result[0], torch.full((2, 1000), 0.5))
        assert torch.allclose(result[1], torch.full((2, 1000), 0.2))
        
        # Verify apply_model was called correctly
        mock_apply_model.assert_called_once()
        args, kwargs = mock_apply_model.call_args
        assert args[0] == mock_model
        assert kwargs['shifts'] == 2
        assert kwargs['split'] is False
        assert kwargs['overlap'] == 0.25 