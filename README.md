# VideoTranscribe

A Python application that automatically transcribes video/audio files using OpenAI Whisper and generates intelligent summaries optimized for Let's Play gaming content using OpenAI's API.

## Features

- **Local Transcription**: Uses OpenAI Whisper for high-quality, local speech-to-text processing
- **AI-Powered Summaries**: Generates contextual episode summaries with OpenAI's API
- **Let's Play Optimized**: Specialized prompts for gaming content with episode context and TODOs
- **Smart Processing**: Automatically skips already-processed files
- **Parallel Processing**: Transcription and summarization run simultaneously for efficiency
- **Multiple Formats**: Supports MP4, MP3, WAV, M4A, MPEG, WEBM, and MPGA files

## Prerequisites

- Python 3.8 or higher
- FFmpeg (required by Whisper for video processing)
- OpenAI API key

### Installing FFmpeg

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd VideoTranscribe
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your-openai-api-key-here
   ```
   
   Get your API key from: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)

## Usage

### Basic Workflow

1. **Create a project folder:**
   ```bash
   mkdir projects/my-game-series
   ```

2. **Add your video/audio files:**
   Place your Let's Play episodes in the project folder. Files are processed in alphabetical order, so name them consistently:
   ```
   projects/my-game-series/
   ├── episode_01.mp4
   ├── episode_02.mp4
   └── episode_03.mp4
   ```

3. **Run the transcription and summarization:**
   ```bash
   python main.py my-game-series
   ```

4. **Find your results:**
   - **Transcripts**: `projects/my-game-series/transcripts/episode_01.txt`
   - **Summaries**: `projects/my-game-series/summaries/episode_01_summary.txt`

### Output Structure
```
projects/my-game-series/
├── episode_01.mp4              # Your original video
├── episode_02.mp4
├── transcripts/                # Generated transcripts
│   ├── episode_01.txt
│   └── episode_02.txt
└── summaries/                  # AI-generated summaries
    ├── episode_01_summary.txt
    └── episode_02_summary.txt
```

## Command Line Options

### Basic Usage
```bash
python main.py <project_name>
```

### Advanced Options

| Option | Description | Default |
|--------|-------------|---------|
| `--whisper-model` | Whisper model for transcription | `turbo` |
| `--openai-model` | OpenAI model for summarization | `gpt-4o-mini` |
| `--transcribe-only` | Only transcribe (skip summarization) | - |
| `--summarize-only` | Only summarize existing transcripts | - |
| `--no-parallel` | Process sequentially instead of parallel | - |

### Examples

**Use different models:**
```bash
python main.py my-game --whisper-model base --openai-model gpt-4
```

**Only transcribe videos:**
```bash
python main.py my-game --transcribe-only
```

**Only generate summaries for existing transcripts:**
```bash
python main.py my-game --summarize-only
```

**Process files sequentially:**
```bash
python main.py my-game --no-parallel
```

## Available Models

### Whisper Models
| Model | Size | Speed | Accuracy | Memory |
|-------|------|-------|----------|--------|
| `tiny` | 39 MB | Very Fast | Low | ~1 GB |
| `base` | 74 MB | Fast | Fair | ~1 GB |
| `small` | 244 MB | Medium | Good | ~2 GB |
| `medium` | 769 MB | Slow | Better | ~5 GB |
| `large` | 1550 MB | Very Slow | Best | ~10 GB |
| `turbo` | 1550 MB | Fast | Best | ~6 GB |

### OpenAI Models
- `gpt-3.5-turbo` - Fast and cost-effective
- `gpt-4o-mini` - Balanced performance (default)
- `gpt-4o` - High quality
- `gpt-4` - Premium quality
- `gpt-4-turbo` - Latest GPT-4 with optimizations

## Environment Configuration

You can override defaults by setting these variables in your `.env` file:

```bash
# Required
OPENAI_API_KEY=your-api-key-here

# Optional overrides
OPENAI_MODEL=gpt-4                    # Override default model
OPENAI_MAX_TOKENS=1500               # Override token limit (default: 1000)
```

## Summary Format

The AI generates structured summaries for each episode including:

- **Episode Summary**: Overview of what happened
- **Key Events**: Important story moments and achievements  
- **Decisions Made**: Significant choices or strategies
- **TODOs for Future Episodes**: Reminders and objectives
- **Notes**: Technical issues, funny moments, observations

Each summary includes context from previous episodes to maintain continuity across your Let's Play series.

## Troubleshooting

### Common Issues

**"No module named 'whisper'"**
```bash
pip install -r requirements.txt
```

**"[Errno 2] No such file or directory: 'ffmpeg'"**
- Install FFmpeg using the instructions in Prerequisites

**"OPENAI_API_KEY environment variable not set"**
- Copy `.env.example` to `.env` and add your API key

**"Error loading model"**
- Ensure you have enough disk space and memory for the Whisper model
- Try a smaller model like `base` or `small`

### Debug Mode

Enable debug logging by setting `DEBUG = True` in `main.py` to see:
- Full OpenAI API prompts
- Model parameters
- Processing details

## License

This project is open source. See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.