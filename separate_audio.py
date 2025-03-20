import torch
import torchaudio
import numpy as np
import os
import argparse
import time
from pathlib import Path
from demucs.apply import apply_model
from demucs.pretrained import get_model
from demucs.audio import convert_audio
from utils import setup_logger, Config


# Set up logger
logger = setup_logger("audiosplitter")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Audio source separation tool")
    parser.add_argument("--input", type=str, default="vivo.mp3", help="Input audio file path")
    parser.add_argument("--model", type=str, default=None, 
                        choices=["htdemucs", "htdemucs_ft", "mdx_extra", "mdx_extra_q"], 
                        help="Model to use for separation (overrides config)")
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to use (cuda or cpu, overrides config)")
    parser.add_argument("--shifts", type=int, default=None, 
                        help="Number of random shifts for equivariant stabilization (overrides config)")
    parser.add_argument("--overlap", type=float, default=None, 
                        help="Overlap between chunks (0 to 1, overrides config)")
    parser.add_argument("--chunk-size", type=int, default=None, 
                        help="Chunk size in seconds (overrides config)")
    parser.add_argument("--split", action="store_true", 
                        help="Use split separation (overrides config)")
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="Output directory (overrides config)")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to configuration file")
    return parser.parse_args()


def load_config_with_args(args):
    """
    Load configuration and override with command line arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        dict: Final configuration
    """
    try:
        config = Config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except (FileNotFoundError, ValueError) as e:
        logger.warning(f"Failed to load configuration: {e}. Using default values.")
        # Create an empty config object with default values
        config = Config.__new__(Config)
        config.config = {}

    # Get configuration sections
    model_config = config.get_model_config()
    processing_config = config.get_processing_config()
    output_config = config.get_output_config()
    
    # Override model configuration with command line arguments
    if args.model is not None:
        model_config['name'] = args.model
    
    if args.device is not None:
        model_config['device'] = args.device
    elif model_config.get('device') == 'auto':
        model_config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.shifts is not None:
        model_config['shifts'] = args.shifts
    
    if args.split:
        model_config['split'] = True
    
    if args.overlap is not None:
        model_config['overlap'] = args.overlap
    
    # Override processing configuration
    if args.chunk_size is not None:
        processing_config['chunk_size'] = args.chunk_size
    
    # Override output configuration
    if args.output_dir is not None:
        output_config['directory'] = args.output_dir
    
    # Create final configuration dictionary
    final_config = {
        'model': model_config,
        'processing': processing_config,
        'output': output_config,
        'input_file': args.input
    }
    
    return final_config


def load_model(model_name, device):
    """Load the separation model."""
    logger.info(f"Loading model {model_name}...")
    model = get_model(model_name)
    model.to(device)
    model.eval()
    
    # Get the model's maximum supported length
    try:
        training_length = model.training_length
        logger.debug(f"Model training length: {training_length} samples")
        max_length = training_length * 0.98  # Stay slightly below the limit
    except:
        # Default to a safe value (about 6 seconds at 48kHz)
        max_length = 250000
        logger.debug(f"Using default safe length: {max_length} samples")
        
    return model, max_length


def load_and_preprocess_audio(input_file, model):
    """Load and preprocess the audio file."""
    logger.info(f"Loading audio file {input_file}...")
    info = torchaudio.info(input_file)
    logger.info(f"Sample rate: {info.sample_rate} Hz, Duration: {info.num_frames/info.sample_rate:.2f}s")
    
    wav, sr = torchaudio.load(input_file)
    wav = convert_audio(wav, sr, model.samplerate, model.audio_channels)
    
    # Normalize the audio
    wav_mean = wav.mean(dim=1, keepdim=True)
    wav_std = wav.std(dim=1, keepdim=True)
    wav = (wav - wav_mean) / wav_std
    
    return wav, wav_mean, wav_std


def process_chunk(chunk, model, device, shifts, split, overlap, fade_in=None, fade_out=None, chunk_idx=0, start=0, audio_length=0):
    """Process a single audio chunk."""
    # Add batch dimension if needed
    if chunk.dim() == 2:
        chunk = chunk.unsqueeze(0)
    
    # Move to device and process
    with torch.no_grad():
        chunk = chunk.to(device)
        
        # Separation
        chunk_sources = apply_model(
            model, 
            chunk, 
            shifts=shifts,
            split=split,
            overlap=overlap,
            progress=False
        )
        
        # Remove batch dimension
        if chunk_sources.dim() == 4:
            chunk_sources = chunk_sources.squeeze(0)
    
    # Apply fade in/out if provided
    if fade_in is not None or fade_out is not None:
        for src_idx in range(chunk_sources.shape[0]):
            src_chunk = chunk_sources[src_idx]
            
            # Apply fade in (except for first chunk)
            if fade_in is not None and chunk_idx > 0 and start + fade_in.shape[0] <= audio_length:
                fade_in_device = fade_in.to(src_chunk.device)
                src_chunk[:, :fade_in.shape[0]] *= fade_in_device
            
            # Apply fade out (except for last chunk)
            if fade_out is not None and start + chunk.shape[2] < audio_length:
                fade_out_device = fade_out.to(src_chunk.device)
                src_chunk[:, -fade_out.shape[0]:] *= fade_out_device
    
    return chunk_sources


def save_sources(output_sources, source_names, output_dir, input_file, sample_rate):
    """Save separated sources to output files."""
    logger.info(f"Saving separated sources to {output_dir}...")
    input_name = Path(input_file).stem
    
    for src_idx, name in enumerate(source_names):
        output_path = os.path.join(output_dir, f"{input_name}_{name}.wav")
        torchaudio.save(
            output_path, 
            output_sources[src_idx], 
            sample_rate
        )
        logger.info(f"Saved {output_path}")


def main():
    """Main function for audio separation."""
    start_time = time.time()
    
    # Enable memory-efficient operations
    torch.set_grad_enabled(False)
    
    # Parse arguments and load configuration
    args = parse_arguments()
    config = load_config_with_args(args)
    
    # Extract configuration
    model_config = config['model']
    processing_config = config['processing']
    output_config = config['output']
    input_file = config['input_file']
    
    # Create output directory
    output_dir = output_config.get('directory', 'separated')
    os.makedirs(output_dir, exist_ok=True)
    
    # Print configuration
    logger.info(f"Using device: {model_config.get('device', 'cpu')}")
    logger.info(f"Model: {model_config.get('name', 'htdemucs')}")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Chunk size: {processing_config.get('chunk_size', 6)} seconds")
    logger.info(f"Shifts: {model_config.get('shifts', 2)}")
    logger.info(f"Overlap: {model_config.get('overlap', 0.25)}")
    
    # Load model
    model, max_model_length = load_model(
        model_config.get('name', 'htdemucs'), 
        model_config.get('device', 'cpu')
    )
    model_sample_rate = model.samplerate
    
    # Load and preprocess audio
    wav, wav_mean, wav_std = load_and_preprocess_audio(input_file, model)
    audio_length = wav.shape[1]
    
    # Get source names from the model
    source_names = model.sources
    num_sources = len(source_names)
    logger.info(f"Source tracks to separate: {', '.join(source_names)}")
    
    # Create empty tensors for each source to collect results
    output_sources = [torch.zeros_like(wav) for _ in range(num_sources)]
    
    # Setup chunking parameters
    requested_chunk_size = min(int(processing_config.get('chunk_size', 6) * model_sample_rate), int(max_model_length))
    chunk_size = requested_chunk_size
    overlap_size = int(chunk_size * model_config.get('overlap', 0.25))
    step_size = chunk_size - overlap_size
    
    # Calculate number of chunks
    num_chunks = int(np.ceil((audio_length - overlap_size) / step_size))
    logger.info(f"Processing audio in {num_chunks} chunks of {chunk_size/model_sample_rate:.1f}s with {overlap_size/model_sample_rate:.1f}s overlap...")
    
    # Hann window for overlap-add (initially on CPU, will be moved to correct device when needed)
    window = torch.hann_window(overlap_size * 2)
    fade_in = window[:overlap_size]
    fade_out = window[overlap_size:]
    
    # Process each chunk
    for chunk_idx in range(num_chunks):
        # Calculate chunk boundaries
        start = chunk_idx * step_size
        end = min(start + chunk_size, audio_length)
        
        # Print progress
        current_time = start / model_sample_rate
        end_time = end / model_sample_rate
        progress = (chunk_idx + 1) / num_chunks * 100
        logger.info(f"Processing chunk {chunk_idx+1}/{num_chunks} ({current_time:.1f}s - {end_time:.1f}s) - {progress:.1f}% complete")
        
        # Extract chunk
        chunk = wav[:, start:end]
        
        try:
            # Process the chunk
            chunk_sources = process_chunk(
                chunk, 
                model, 
                model_config.get('device', 'cpu'), 
                model_config.get('shifts', 2), 
                model_config.get('split', False), 
                model_config.get('overlap', 0.25),
                fade_in, 
                fade_out, 
                chunk_idx, 
                start, 
                audio_length
            )
            
            # For each source, add to output
            for src_idx in range(num_sources):
                # Move back to CPU for storage
                src_chunk_cpu = chunk_sources[src_idx].cpu()
                
                # Add to output using overlap-add
                output_sources[src_idx][:, start:end] += src_chunk_cpu
        
        except ValueError as e:
            if "longer than training length" in str(e):
                logger.warning(f"Error: Chunk too large. Retrying with smaller chunk...")
                # Reduce chunk size for future chunks
                chunk_size = int(chunk_size * 0.8)
                overlap_size = int(chunk_size * model_config.get('overlap', 0.25))
                step_size = chunk_size - overlap_size
                
                # Recalculate window sizes
                window = torch.hann_window(overlap_size * 2)
                fade_in = window[:overlap_size]
                fade_out = window[overlap_size:]
                
                # Recalculate number of chunks
                remaining_samples = audio_length - start
                remaining_chunks = int(np.ceil((remaining_samples - overlap_size) / step_size))
                num_chunks = chunk_idx + remaining_chunks
                
                logger.info(f"Adjusted to {remaining_chunks} more chunks of {chunk_size/model_sample_rate:.1f}s...")
                
                # Try again with the same chunk index (will now use a different size)
                chunk_idx -= 1
                continue
            else:
                logger.error(f"Error processing chunk: {e}")
                raise e
        
        # Free up memory
        del chunk_sources
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Denormalize the results
    for src_idx in range(num_sources):
        output_sources[src_idx] = output_sources[src_idx] * wav_std + wav_mean
    
    # Save sources
    save_sources(output_sources, source_names, output_dir, input_file, model_sample_rate)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Separation complete! Files saved in {output_dir}")
    logger.info(f"Separated tracks: {', '.join(source_names)}")
    logger.info(f"Total processing time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")


if __name__ == "__main__":
    main()