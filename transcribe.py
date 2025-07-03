#!/usr/bin/env python3
"""
VideoTranscribe - Local Video/Audio Transcription Tool
Uses OpenAI Whisper for local speech-to-text processing.
"""

import argparse
import os
import sys
from pathlib import Path
import whisper

# Configuration Constants
DEFAULT_MODEL = "turbo"
AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large", "turbo"]
SUPPORTED_EXTENSIONS = {".mp4", ".mpeg", ".webm", ".m4a", ".mp3", ".mpga", ".wav"}
PROJECTS_DIR = "projects"
TRANSCRIPTS_DIR = "transcripts"
TRANSCRIPT_EXTENSION = ".txt"
ENCODING = "utf-8"


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Transcribe video/audio files using OpenAI Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python transcribe.py my_project
  python transcribe.py my_project --model base
  python transcribe.py lecture_series --model large

Available models: {', '.join(AVAILABLE_MODELS)}
Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}
        """
    )
    
    parser.add_argument(
        "project_name",
        help="Name of the project folder containing media files"
    )
    
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=AVAILABLE_MODELS,
        help=f"Whisper model to use (default: {DEFAULT_MODEL})"
    )
    
    return parser.parse_args()


def validate_project_structure(project_name):
    """Validate and create project directory structure."""
    project_path = Path(PROJECTS_DIR) / project_name
    transcripts_path = project_path / TRANSCRIPTS_DIR
    
    if not project_path.exists():
        print(f"Error: Project folder '{project_path}' does not exist.")
        print(f"Please create it and add your media files.")
        sys.exit(1)
    
    if not project_path.is_dir():
        print(f"Error: '{project_path}' is not a directory.")
        sys.exit(1)
    
    # Create transcripts directory if it doesn't exist
    transcripts_path.mkdir(exist_ok=True)
    
    return project_path, transcripts_path


def discover_media_files(project_path):
    """Discover and sort media files in the project directory."""
    media_files = []
    
    for file_path in project_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            media_files.append(file_path)
    
    # Sort alphabetically
    media_files.sort(key=lambda x: x.name.lower())
    
    return media_files


def get_transcript_path(media_file, transcripts_path):
    """Get the transcript file path for a given media file."""
    transcript_name = media_file.stem + TRANSCRIPT_EXTENSION
    return transcripts_path / transcript_name


def transcript_exists(media_file, transcripts_path):
    """Check if transcript already exists for the media file."""
    transcript_path = get_transcript_path(media_file, transcripts_path)
    return transcript_path.exists()


def load_whisper_model(model_name):
    """Load the Whisper model."""
    try:
        print(f"Loading Whisper model: {model_name}")
        model = whisper.load_model(model_name)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")
        sys.exit(1)


def transcribe_file(model, media_file, transcript_path):
    """Transcribe a single media file."""
    try:
        print(f"Processing: {media_file.name}")
        
        # Transcribe with progress bar
        result = model.transcribe(str(media_file), verbose=False)
        
        # Save transcript
        with open(transcript_path, 'w', encoding=ENCODING) as f:
            f.write(result["text"])
        
        print(f"Transcript saved: {transcript_path.name}")
        return True
        
    except Exception as e:
        print(f"Error transcribing {media_file.name}: {e}")
        return False


def main():
    """Main application entry point."""
    args = parse_arguments()
    
    print(f"VideoTranscribe - Processing project: {args.project_name}")
    print(f"Using model: {args.model}")
    print("-" * 50)
    
    # Validate project structure
    project_path, transcripts_path = validate_project_structure(args.project_name)
    
    # Discover media files
    media_files = discover_media_files(project_path)
    
    if not media_files:
        print(f"No media files found in {project_path}")
        print(f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        sys.exit(0)
    
    print(f"Found {len(media_files)} media files")
    
    # Check for existing transcripts
    files_to_process = []
    skipped_files = []
    
    for media_file in media_files:
        if transcript_exists(media_file, transcripts_path):
            skipped_files.append(media_file)
        else:
            files_to_process.append(media_file)
    
    if skipped_files:
        print(f"Skipping {len(skipped_files)} files (transcripts already exist):")
        for skipped_file in skipped_files:
            print(f"  - {skipped_file.name}")
    
    if not files_to_process:
        print("All files have been processed. No work to do.")
        sys.exit(0)
    
    print(f"Processing {len(files_to_process)} files...")
    print("-" * 50)
    
    # Load Whisper model
    model = load_whisper_model(args.model)
    
    # Process files
    successful = 0
    failed = 0
    
    for i, media_file in enumerate(files_to_process, 1):
        print(f"\nFile {i} of {len(files_to_process)}")
        transcript_path = get_transcript_path(media_file, transcripts_path)
        
        if transcribe_file(model, media_file, transcript_path):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print("-" * 50)
    print(f"Processing complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {len(skipped_files)}")
    
    # TODO: Phase 2 - Add post-processing of transcripts
    # TODO: Phase 2 - Add transcript analysis and enhancement features


if __name__ == "__main__":
    main()