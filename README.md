# VideoSummarize

A Python application that automatically transcribes video/audio files using OpenAI Whisper and generates intelligent summaries content using OpenAI's API. The example prompt is optimized for Let's Play gaming content, but it can be modified to fit any type of video.

Created by Martin 'quill18' Glaude, mostly using Claude Code.

## Features

- **Local Transcription**: Uses OpenAI Whisper for high-quality, local speech-to-text processing (this is done on your local machine, for free)
- **AI-Powered Summaries**: Generates contextual episode summaries with OpenAI's API (this requires an API key) TODO: Add support for local LLMs / OpenRouter / etc...
- **Smart Processing**: Automatically skips already-processed files
- **Parallel Processing**: Transcription and summarization run simultaneously for better speed
- **Multiple Formats**: Supports MP4, MP3, WAV, M4A, MPEG, WEBM, and MPGA files

## Prerequisites

- Python 3.8 or higher
- FFmpeg (required by Whisper for video processing)
- OpenAI API key (for summarizing)

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
   - **Summaries**: `projects/my-game-series/summaries/episode_01_summary.md`

### Output Structure
```
projects/my-game-series/
├── episode_01.mp4              # Your original video
├── episode_02.mp4
├── transcripts/                # Generated transcripts
│   ├── episode_01.txt
│   └── episode_02.txt
└── summaries/                  # AI-generated summaries
    ├── episode_01_summary.md
    └── episode_02_summary.md
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

## Available Whisper Models

| Model | Size | Speed | Accuracy | Memory |
|-------|------|-------|----------|--------|
| `tiny` | 39 MB | Very Fast | Low | ~1 GB |
| `base` | 74 MB | Fast | Fair | ~1 GB |
| `small` | 244 MB | Medium | Good | ~2 GB |
| `medium` | 769 MB | Slow | Better | ~5 GB |
| `large` | 1550 MB | Very Slow | Best | ~10 GB |
| `turbo` | 1550 MB | Fast | Best | ~6 GB |

Turbo is the default and is recommended if you have sufficient memory.

## Environment Configuration

You can override defaults by setting these variables in your `.env` file:

```bash
# Required
OPENAI_API_KEY=your-api-key-here

# Optional overrides
WHISPER_MODEL=base                   # Override default Whisper model (default: turbo)
OPENAI_MODEL=gpt-4                   # Override default OpenAI model (default: gpt-4o-mini)
OPENAI_MAX_TOKENS=1500               # Override token limit (default: 1000)
```

## Customizing AI Prompts

You can customize the AI summarization prompts to better suit your content style:

1. **Create your custom prompt file:**
   ```bash
   cp prompts/default_system_prompt.txt prompts/system_prompt_override.txt
   ```
   
   **Important:** You must create this file yourself. It's not included in the repository to prevent `git pull` from overwriting your customizations.

2. **Edit the override file** with your preferred prompt style:
   ```bash
   nano prompts/system_prompt_override.txt
   ```

3. **How it works:**
   - If `prompts/system_prompt_override.txt` exists and has content, it will be used
   - If the override file is missing or empty, the default prompt is used
   - The override file is git-ignored, so your customizations are safe from future updates

**Example customizations:**
- Add specific game terminology or context
- Adjust the summary format or sections
- Include additional analysis categories
- Modify the tone or style of summaries

### Project-Specific Prompt Extensions

For individual projects, you can add custom context to the AI prompts on a per-project basis:

1. **Create a project-specific extension file:**
   ```bash
   # Inside your project folder
   echo "Additional context for this game..." > projects/my-game-series/system_prompt_extension.txt
   ```

2. **Add project-specific instructions:**
   ```bash
   # Example for Terra Invicta project
   nano projects/Terra_Invicta/system_prompt_extension.txt
   ```
   
   Example content:
   ```
   This is a complex grand strategy game about defending Earth from alien invasion. 
   Focus on faction relationships, research priorities, and strategic decisions.
   Pay attention to diplomatic choices and their long-term consequences.
   ```

3. **How it works:**
   - The extension content is appended to the main system prompt
   - Only applies to that specific project
   - File is git-ignored so it won't be committed to the repository
   - If the file doesn't exist, the standard prompt is used

**Use cases:**
- Add game-specific terminology and mechanics
- Include character names and relationships
- Set context for complex storylines
- Adjust focus areas (combat, story, exploration, etc.)
- Add reminders about specific gameplay mechanics

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
