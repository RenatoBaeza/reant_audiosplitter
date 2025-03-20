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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Audio source separation tool")
    parser.add_argument("--input", type=str, default="vivo.mp3", help="Input audio file path")
    parser.add_argument("--model", type=str, default="htdemucs", 
                        choices=["htdemucs", "htdemucs_ft", "mdx_extra", "mdx_extra_q"], 
                        help="Model to use for separation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--shifts", type=int, default=2, 
                        help="Number of random shifts for equivariant stabilization")
    parser.add_argument("--overlap", type=float, default=0.25, 
                        help="Overlap between chunks (0 to 1)")
    parser.add_argument("--chunk-size", type=int, default=6, 
                        help="Chunk size in seconds")
    parser.add_argument("--split", action="store_true", 
                        help="Use split separation")
    parser.add_argument("--output_dir", type=str, default="separated", 
                        help="Output directory")
    return parser.parse_args()


def load_model(model_name, device):
    """Load the separation model."""
    print(f"Loading model {model_name}...")
    model = get_model(model_name)
    model.to(device)
    model.eval()
    
    # Get the model's maximum supported length
    try:
        training_length = model.training_length
        print(f"Model training length: {training_length} samples")
        max_length = training_length * 0.98  # Stay slightly below the limit
    except:
        # Default to a safe value (about 6 seconds at 48kHz)
        max_length = 250000
        print(f"Using default safe length: {max_length} samples")
        
    return model, max_length


def load_and_preprocess_audio(input_file, model):
    """Load and preprocess the audio file."""
    print("Loading audio file...")
    info = torchaudio.info(input_file)
    print(f"Sample rate: {info.sample_rate} Hz, Duration: {info.num_frames/info.sample_rate:.2f}s")
    
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
    print("Saving separated sources...")
    input_name = Path(input_file).stem
    
    for src_idx, name in enumerate(source_names):
        output_path = os.path.join(output_dir, f"{input_name}_{name}.wav")
        torchaudio.save(
            output_path, 
            output_sources[src_idx], 
            sample_rate
        )
        print(f"Saved {output_path}")


def main():
    """Main function for audio separation."""
    start_time = time.time()
    
    # Enable memory-efficient operations
    torch.set_grad_enabled(False)
    
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration
    print(f"Using device: {args.device}")
    print(f"Model: {args.model}")
    print(f"Input file: {args.input}")
    print(f"Chunk size: {args.chunk_size} seconds")
    print(f"Shifts: {args.shifts}")
    print(f"Overlap: {args.overlap}")
    
    # Load model
    model, max_model_length = load_model(args.model, args.device)
    model_sample_rate = model.samplerate
    
    # Load and preprocess audio
    wav, wav_mean, wav_std = load_and_preprocess_audio(args.input, model)
    audio_length = wav.shape[1]
    
    # Get source names from the model
    source_names = model.sources
    num_sources = len(source_names)
    print(f"Source tracks to separate: {', '.join(source_names)}")
    
    # Create empty tensors for each source to collect results
    output_sources = [torch.zeros_like(wav) for _ in range(num_sources)]
    
    # Setup chunking parameters
    requested_chunk_size = min(int(args.chunk_size * model_sample_rate), int(max_model_length))
    chunk_size = requested_chunk_size
    overlap_size = int(chunk_size * args.overlap)
    step_size = chunk_size - overlap_size
    
    # Calculate number of chunks
    num_chunks = int(np.ceil((audio_length - overlap_size) / step_size))
    print(f"Processing audio in {num_chunks} chunks of {chunk_size/model_sample_rate:.1f}s with {overlap_size/model_sample_rate:.1f}s overlap...")
    
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
        print(f"Processing chunk {chunk_idx+1}/{num_chunks} ({current_time:.1f}s - {end_time:.1f}s) - {progress:.1f}% complete")
        
        # Extract chunk
        chunk = wav[:, start:end]
        
        try:
            # Process the chunk
            chunk_sources = process_chunk(
                chunk, 
                model, 
                args.device, 
                args.shifts, 
                args.split, 
                args.overlap,
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
                print(f"Error: Chunk too large. Retrying with smaller chunk...")
                # Reduce chunk size for future chunks
                chunk_size = int(chunk_size * 0.8)
                overlap_size = int(chunk_size * args.overlap)
                step_size = chunk_size - overlap_size
                
                # Recalculate window sizes
                window = torch.hann_window(overlap_size * 2)
                fade_in = window[:overlap_size]
                fade_out = window[overlap_size:]
                
                # Recalculate number of chunks
                remaining_samples = audio_length - start
                remaining_chunks = int(np.ceil((remaining_samples - overlap_size) / step_size))
                num_chunks = chunk_idx + remaining_chunks
                
                print(f"Adjusted to {remaining_chunks} more chunks of {chunk_size/model_sample_rate:.1f}s...")
                
                # Try again with the same chunk index (will now use a different size)
                chunk_idx -= 1
                continue
            else:
                raise e
        
        # Free up memory
        del chunk_sources
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Denormalize the results
    for src_idx in range(num_sources):
        output_sources[src_idx] = output_sources[src_idx] * wav_std + wav_mean
    
    # Save sources
    save_sources(output_sources, source_names, args.output_dir, args.input, model_sample_rate)
    
    elapsed_time = time.time() - start_time
    print(f"Separation complete! Files saved in {args.output_dir}")
    print(f"Separated tracks: {', '.join(source_names)}")
    print(f"Total processing time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")


if __name__ == "__main__":
    main()