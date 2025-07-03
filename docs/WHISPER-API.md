# OpenAI Whisper API Documentation

## Overview

OpenAI Whisper is a robust speech recognition system that can transcribe audio in multiple languages, translate speech to English, and identify spoken languages. It's available both as an open-source library and through OpenAI's API service.

## Two Ways to Use Whisper

### 1. OpenAI API (Hosted Service)
- **Models**: `whisper-1`, `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`
- **Pricing**: $0.006 per minute of audio
- **File Size Limit**: 25 MB maximum
- **Streaming**: Available for newer models (gpt-4o variants)

### 2. Open Source Library
- **Installation**: `pip install -U openai-whisper`
- **Multiple Models**: tiny, base, small, medium, large, turbo
- **Local Processing**: No API calls required
- **No File Size Limits**: Limited only by system resources

## Supported Audio Formats

Both versions support: `m4a`, `mp3`, `mp4`, `mpeg`, `mpga`, `wav`, `webm`

## OpenAI API Usage

### Setup
```python
from openai import OpenAI
import os

# Initialize client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

### Transcription (Same Language)
```python
from openai import OpenAI

client = OpenAI()

# Using gpt-4o-transcribe (recommended for highest quality)
with open("audio.mp3", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="gpt-4o-transcribe",
        file=audio_file,
        language="en"  # Optional: specify language
    )
    print(transcript.text)

# Using whisper-1 (more format options)
with open("audio.mp3", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="verbose_json",  # json, text, srt, verbose_json, vtt
        timestamp_granularities=["word"]  # word-level timestamps
    )
    print(transcript.text)
```

### Translation (To English)
```python
from openai import OpenAI

client = OpenAI()

# Translation only supports whisper-1
with open("spanish_audio.mp3", "rb") as audio_file:
    translation = client.audio.translations.create(
        model="whisper-1",
        file=audio_file
    )
    print(translation.text)
```

### Streaming Transcription (New)
```python
from openai import OpenAI

client = OpenAI()

# Stream transcription results as they're processed
with open("audio.mp3", "rb") as audio_file:
    stream = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=audio_file,
        response_format="text",
        stream=True
    )
    
    for event in stream:
        print(event)
```

### Advanced Parameters
```python
# For whisper-1 with full parameters
transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file,
    language="en",
    response_format="verbose_json",  # json, text, srt, verbose_json, vtt
    temperature=0.2,         # 0-1, higher = more random
    prompt="Custom context for better accuracy",
    timestamp_granularities=["word", "segment"]  # word and segment timestamps
)

# For gpt-4o models (limited parameters)
transcript = client.audio.transcriptions.create(
    model="gpt-4o-transcribe",
    file=audio_file,
    response_format="json",  # json or text only
    prompt="The following is a technical discussion about AI."
)
```

## Open Source Library Usage

### Installation
```bash
pip install -U openai-whisper
```

### Command Line
```bash
# Basic transcription
whisper audio.mp3 --model turbo

# Specify language and task
whisper japanese.wav --model medium --language Japanese --task translate

# Output formats
whisper audio.mp3 --output_format srt --output_dir ./transcripts
```

### Python API
```python
import whisper

# Load model (downloads on first use)
model = whisper.load_model("turbo")

# Transcribe audio
result = model.transcribe("audio.mp3")
print(result["text"])

# Access detailed results
print(result["segments"])  # Timestamped segments
print(result["language"])  # Detected language
```

### Available Models

| Model  | Size | VRAM | Speed | Accuracy |
|--------|------|------|-------|----------|
| tiny   | 39 MB | ~1 GB | Very Fast | Low |
| base   | 74 MB | ~1 GB | Fast | Fair |
| small  | 244 MB | ~2 GB | Medium | Good |
| medium | 769 MB | ~5 GB | Slow | Better |
| large  | 1550 MB | ~10 GB | Very Slow | Best |
| turbo  | 1550 MB | ~6 GB | Fast | Best |

### Model Loading Options
```python
# Load specific model
model = whisper.load_model("turbo")

# Load with device specification
model = whisper.load_model("medium", device="cuda")  # GPU
model = whisper.load_model("small", device="cpu")    # CPU only
```

## Advanced Features

### Timestamp Granularities (API)
```python
# Word-level timestamps with whisper-1
transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file,
    response_format="verbose_json",
    timestamp_granularities=["word"]
)

# Access word-level timestamps
for word in transcript.words:
    print(f"[{word['start']:.2f}s -> {word['end']:.2f}s] {word['word']}")

# Segment-level timestamps
transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file,
    response_format="verbose_json",
    timestamp_granularities=["segment"]
)

for segment in transcript.segments:
    print(f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}")
```

### Segment-Level Timestamps (Open Source)
```python
result = model.transcribe("audio.mp3", word_timestamps=True)

for segment in result["segments"]:
    print(f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}")
```

### Language Detection
```python
# Auto-detect language (Open Source)
result = model.transcribe("audio.mp3")
print(f"Detected language: {result['language']}")

# Force specific language (Open Source)
result = model.transcribe("audio.mp3", language="es")
```

### Custom Prompts for Better Accuracy
```python
# API - GPT-4o models
transcript = client.audio.transcriptions.create(
    model="gpt-4o-transcribe",
    file=audio_file,
    prompt="The following conversation is about OpenAI, GPT-4, and AI developments."
)

# API - whisper-1 model
transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file,
    prompt="ZyntriQix, Digique Plus, CynapseFive, VortiQore V8"  # Limited to 224 tokens
)

# Open Source
result = model.transcribe(
    "audio.mp3", 
    initial_prompt="This is a technical discussion about machine learning algorithms."
)
```

### Realtime Streaming (API)
```python
# WebSocket connection for real-time transcription
import websockets
import json
import base64

async def realtime_transcription():
    uri = "wss://api.openai.com/v1/realtime?intent=transcription"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    async with websockets.connect(uri, extra_headers=headers) as websocket:
        # Setup session
        session_config = {
            "type": "transcription_session.update",
            "input_audio_format": "pcm16",
            "input_audio_transcription": {
                "model": "gpt-4o-transcribe",
                "prompt": "",
                "language": ""
            },
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 500
            }
        }
        
        await websocket.send(json.dumps(session_config))
        
        # Stream audio data
        audio_data = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(audio_bytes).decode()
        }
        
        await websocket.send(json.dumps(audio_data))
        
        # Receive transcription events
        async for message in websocket:
            event = json.loads(message)
            if event["type"] == "input_audio_buffer.speech_started":
                print("Speech detected")
            elif event["type"] == "input_audio_buffer.speech_stopped":
                print("Speech ended")
            elif event["type"] == "input_audio_buffer.committed":
                print(f"Transcription: {event['transcript']}")
```

## Best Practices

### Audio Quality
- Use high-quality audio (16kHz+ sample rate)
- Minimize background noise
- Ensure clear speech without overlapping speakers

### Performance Optimization
- Use `turbo` model for best speed/accuracy balance
- Process longer audio files in chunks if memory is limited
- Use GPU acceleration when available

### Error Handling
```python
try:
    result = model.transcribe("audio.mp3")
except Exception as e:
    print(f"Transcription failed: {e}")
```

## Common Use Cases

1. **Meeting Transcription**: Convert recorded meetings to text
2. **Video Subtitles**: Generate subtitles for video content
3. **Voice Notes**: Transcribe personal voice memos
4. **Language Learning**: Transcribe foreign language content
5. **Accessibility**: Convert audio content for hearing-impaired users

## Supported Languages

Both API and open-source versions support the following languages:

Afrikaans, Arabic, Armenian, Azerbaijani, Belarusian, Bosnian, Bulgarian, Catalan, Chinese, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, Galician, German, Greek, Hebrew, Hindi, Hungarian, Icelandic, Indonesian, Italian, Japanese, Kannada, Kazakh, Korean, Latvian, Lithuanian, Macedonian, Malay, Marathi, Maori, Nepali, Norwegian, Persian, Polish, Portuguese, Romanian, Russian, Serbian, Slovak, Slovenian, Spanish, Swahili, Swedish, Tagalog, Tamil, Thai, Turkish, Ukrainian, Urdu, Vietnamese, and Welsh.

**Note**: Only languages with <50% word error rate are listed. The model will work on other languages but with lower quality.

## Handling Large Files

For files larger than 25MB (API limit), you can split them using PyDub:

```python
from pydub import AudioSegment

# Load audio file
song = AudioSegment.from_mp3("large_audio.mp3")

# Split into 10-minute chunks
ten_minutes = 10 * 60 * 1000  # PyDub works in milliseconds
chunks = []

for i in range(0, len(song), ten_minutes):
    chunk = song[i:i + ten_minutes]
    chunk_filename = f"chunk_{i//ten_minutes}.mp3"
    chunk.export(chunk_filename, format="mp3")
    chunks.append(chunk_filename)

# Transcribe each chunk
full_transcript = ""
for chunk_file in chunks:
    with open(chunk_file, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=audio_file
        )
        full_transcript += transcript.text + " "

print(full_transcript)
```

## Post-Processing with GPT-4

Improve transcription accuracy by post-processing with GPT-4:

```python
def improve_transcript_with_gpt4(transcript, context_prompt):
    system_prompt = f"""
    You are a helpful assistant for transcription correction. Your task is to correct 
    any spelling discrepancies in the transcribed text. {context_prompt}
    Only add necessary punctuation such as periods, commas, and capitalization, 
    and use only the context provided.
    """
    
    response = client.chat.completions.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript}
        ]
    )
    
    return response.choices[0].message.content

# Usage
raw_transcript = "transcribe audio here..."
context = "Make sure technical terms like 'API', 'JSON', 'WebSocket' are spelled correctly."
corrected_transcript = improve_transcript_with_gpt4(raw_transcript, context)
```

## Limitations

- **API Rate Limits**: Check OpenAI's current rate limits
- **File Size**: 25MB limit for API, no limit for open source
- **Streaming**: Available for gpt-4o models, not whisper-1
- **Parameter Support**: gpt-4o models have limited parameters (no timestamp_granularities, verbose_json)
- **Translation**: Only whisper-1 supports translation to English
- **Accuracy**: Performance varies with audio quality and accent
- **Languages**: Best performance on English, good on other major languages

## Security Considerations

- Never expose API keys in client-side code
- Store API keys securely using environment variables
- Be aware that API requests send audio to OpenAI servers
- Use open source version for sensitive content that must stay local

## Error Codes and Troubleshooting

### Common API Errors
- `400 Bad Request`: Invalid audio format or file too large
- `401 Unauthorized`: Invalid API key
- `429 Too Many Requests`: Rate limit exceeded

### Open Source Issues
- `RuntimeError`: Usually indicates insufficient VRAM
- `FileNotFoundError`: Audio file path is incorrect
- `ModuleNotFoundError`: Missing dependencies (install with `pip install -U openai-whisper`)

## Resources

- [OpenAI Whisper GitHub](https://github.com/openai/whisper)
- [OpenAI API Documentation](https://platform.openai.com/docs/guides/speech-to-text)
- [OpenAI Pricing](https://openai.com/pricing)