# reant_audiosplitter

A Python tool for splitting audio tracks into separate instrument and vocal components using deep learning models.

## Features

- Separates audio files into multiple stems (vocals, drums, bass, other)
- Supports various Demucs model variants
- Processes audio in chunks to handle files of any length
- GPU acceleration (when available)
- Configurable processing parameters

## Requirements

- Python 3.8+
- PyTorch
- Torchaudio
- Demucs
- NumPy

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/reant_audiosplitter.git
   cd reant_audiosplitter
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the script with your audio file:

```
python separate_audio.py --input your_audio_file.mp3
```

### Options

- `--input`: Input audio file path (default: vivo.mp3)
- `--model`: Model to use for separation (choices: htdemucs, htdemucs_ft, mdx_extra, mdx_extra_q, default: htdemucs)
- `--device`: Device to use (cuda or cpu, default: cuda if available)
- `--shifts`: Number of random shifts for equivariant stabilization (default: 2)
- `--overlap`: Overlap between chunks (0 to 1, default: 0.25)
- `--chunk-size`: Chunk size in seconds (default: 6)
- `--split`: Use split separation
- `--output_dir`: Output directory (default: separated)

## Testing

The project includes a comprehensive test suite using pytest. To run the tests:

```
pytest
```

To run tests with coverage report:

```
pytest --cov=. tests/
```

The test suite includes:
- Unit tests for configuration management
- Tests for audio processing functionality
- Logger and utility function tests

## Output

Separated tracks will be saved in the specified output directory with file names following the pattern: `{input_filename}_{stem_name}.wav`