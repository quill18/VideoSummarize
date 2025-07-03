#!/usr/bin/env python3
"""
VideoTranscribe - Phase 2: Local Video/Audio Transcription + AI Summarization
Uses OpenAI Whisper for local speech-to-text processing and OpenAI API for Let's Play summaries.
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import threading
import queue
import time

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Debug Configuration
DEBUG = True

# Whisper Configuration
DEFAULT_WHISPER_MODEL = "turbo"
AVAILABLE_WHISPER_MODELS = ["tiny", "base", "small", "medium", "large", "turbo"]
SUPPORTED_EXTENSIONS = {".mp4", ".mpeg", ".webm", ".m4a", ".mp3", ".mpga", ".wav"}

# OpenAI Configuration
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
AVAILABLE_OPENAI_MODELS = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini"]
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "1000"))
OPENAI_TEMPERATURE = 0.7

# Directory Structure
PROJECTS_DIR = "projects"
TRANSCRIPTS_DIR = "transcripts"
SUMMARIES_DIR = "summaries"
TRANSCRIPT_EXTENSION = ".txt"
SUMMARY_EXTENSION = "_summary.txt"
ENCODING = "utf-8"

# Let's Play Prompt Template
LETS_PLAY_SYSTEM_PROMPT = """You are an AI assistant specialized in summarizing Let's Play gaming videos for content creators. 

Your task is to analyze transcripts of Let's Play episodes and create helpful summaries that will assist the gamer in future recording sessions.

For each episode, provide:
1. **Episode Summary**: A concise overview of what happened in this episode
2. **Key Events**: Important story moments, achievements, or gameplay milestones
3. **Decisions Made**: Any significant choices or strategies employed
4. **TODOs for Future Episodes**: Specific reminders, objectives, or things to remember for upcoming sessions
5. **Notes**: Any technical issues, funny moments, or other relevant observations

Keep summaries focused and practical - they should serve as useful reference material for the creator when planning future episodes."""

LETS_PLAY_PROMPT_TEMPLATE = """Game: {game_name}
Episode: {episode_number}
Project: {project_name}

{previous_context}

Current Episode Transcript:
{transcript}

Please provide a structured summary following the format specified in the system prompt."""

CONTEXT_TEMPLATE = """Previous Episode Context:
{previous_summaries}

---"""

# Threading Configuration
MAX_WORKER_THREADS = 2
QUEUE_TIMEOUT = 1.0


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="VideoTranscribe - Transcribe and summarize Let's Play videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python main.py my_game_project
  python main.py my_game_project --whisper-model base --openai-model gpt-4
  python main.py my_game_project --transcribe-only
  python main.py my_game_project --summarize-only

Available Whisper models: {', '.join(AVAILABLE_WHISPER_MODELS)}
Available OpenAI models: {', '.join(AVAILABLE_OPENAI_MODELS)}
Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}
        """
    )
    
    parser.add_argument(
        "project_name",
        help="Name of the project folder containing media files"
    )
    
    parser.add_argument(
        "--whisper-model",
        default=DEFAULT_WHISPER_MODEL,
        choices=AVAILABLE_WHISPER_MODELS,
        help=f"Whisper model to use (default: {DEFAULT_WHISPER_MODEL})"
    )
    
    parser.add_argument(
        "--openai-model",
        default=DEFAULT_OPENAI_MODEL,
        choices=AVAILABLE_OPENAI_MODELS,
        help=f"OpenAI model to use for summarization (default: {DEFAULT_OPENAI_MODEL})"
    )
    
    parser.add_argument(
        "--transcribe-only",
        action="store_true",
        help="Only perform transcription (skip summarization)"
    )
    
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Only perform summarization (skip transcription)"
    )
    
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing (process sequentially)"
    )
    
    return parser.parse_args()


def validate_environment():
    """Validate that required environment variables are set."""
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please copy .env.example to .env and add your OpenAI API key.")
        sys.exit(1)


def validate_project_structure(project_name):
    """Validate and create project directory structure."""
    project_path = Path(PROJECTS_DIR) / project_name
    transcripts_path = project_path / TRANSCRIPTS_DIR
    summaries_path = project_path / SUMMARIES_DIR
    
    if not project_path.exists():
        print(f"Error: Project folder '{project_path}' does not exist.")
        print(f"Please create it and add your media files.")
        sys.exit(1)
    
    if not project_path.is_dir():
        print(f"Error: '{project_path}' is not a directory.")
        sys.exit(1)
    
    # Create transcripts and summaries directories if they don't exist
    transcripts_path.mkdir(exist_ok=True)
    summaries_path.mkdir(exist_ok=True)
    
    return project_path, transcripts_path, summaries_path


def discover_media_files(project_path):
    """Discover and sort media files in the project directory."""
    media_files = []
    
    for file_path in project_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            media_files.append(file_path)
    
    # Sort alphabetically for consistent episode ordering
    media_files.sort(key=lambda x: x.name.lower())
    
    return media_files


def create_constants_dict():
    """Create a dictionary of all constants to pass to modules."""
    return {
        'DEBUG': DEBUG,
        'SUPPORTED_EXTENSIONS': SUPPORTED_EXTENSIONS,
        'TRANSCRIPTS_DIR': TRANSCRIPTS_DIR,
        'SUMMARIES_DIR': SUMMARIES_DIR,
        'TRANSCRIPT_EXTENSION': TRANSCRIPT_EXTENSION,
        'SUMMARY_EXTENSION': SUMMARY_EXTENSION,
        'ENCODING': ENCODING,
        'OPENAI_MAX_TOKENS': OPENAI_MAX_TOKENS,
        'OPENAI_TEMPERATURE': OPENAI_TEMPERATURE,
        'LETS_PLAY_SYSTEM_PROMPT': LETS_PLAY_SYSTEM_PROMPT,
        'LETS_PLAY_PROMPT_TEMPLATE': LETS_PLAY_PROMPT_TEMPLATE,
        'CONTEXT_TEMPLATE': CONTEXT_TEMPLATE,
    }


def process_pipeline(args):
    """Main processing pipeline orchestrator."""
    # Validate environment and project
    if not args.transcribe_only:
        validate_environment()
    
    project_path, transcripts_path, summaries_path = validate_project_structure(args.project_name)
    
    # Discover media files
    media_files = discover_media_files(project_path)
    
    if not media_files:
        print(f"No media files found in {project_path}")
        print(f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        return
    
    print(f"Found {len(media_files)} media files")
    
    # Create constants dictionary
    constants = create_constants_dict()
    
    # Import modules (late import to avoid circular dependencies)
    try:
        import transcribe_module
        if not args.transcribe_only:
            import summarize_module
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Make sure all required modules are present.")
        sys.exit(1)
    
    # Process based on mode
    if args.summarize_only:
        # Summarize existing transcripts
        print("Summarize-only mode: Processing existing transcripts")
        summarize_module.summarize_existing_transcripts(
            project_path, transcripts_path, summaries_path, 
            args.project_name, args.openai_model, constants
        )
    elif args.transcribe_only:
        # Transcribe only
        print("Transcribe-only mode: Processing media files")
        transcribe_module.transcribe_media_files(
            media_files, transcripts_path, args.whisper_model, constants
        )
    else:
        # Full pipeline
        if args.no_parallel:
            print("Sequential processing mode")
            # Process sequentially
            transcribe_module.transcribe_media_files(
                media_files, transcripts_path, args.whisper_model, constants
            )
            summarize_module.summarize_existing_transcripts(
                project_path, transcripts_path, summaries_path, 
                args.project_name, args.openai_model, constants
            )
        else:
            print("Parallel processing mode")
            # Process with threading
            process_parallel_pipeline(
                media_files, project_path, transcripts_path, summaries_path,
                args.project_name, args.whisper_model, args.openai_model, constants
            )


def analyze_file_status(media_files, transcripts_path, summaries_path, constants):
    """Analyze which files need transcription and/or summarization."""
    files_status = {
        'need_transcription': [],
        'need_summarization': [],
        'fully_processed': [],
        'skipped_transcription': [],
        'skipped_summarization': []
    }
    
    import transcribe_module
    import summarize_module
    
    for media_file in media_files:
        has_transcript = transcribe_module.transcript_exists(media_file, transcripts_path, constants)
        has_summary = summarize_module.summary_exists(media_file, summaries_path, constants)
        
        if not has_transcript:
            files_status['need_transcription'].append(media_file)
            # If no transcript, we'll need summarization later
            files_status['need_summarization'].append(media_file)
        elif not has_summary:
            files_status['skipped_transcription'].append(media_file)
            files_status['need_summarization'].append(media_file)
        else:
            files_status['fully_processed'].append(media_file)
            files_status['skipped_transcription'].append(media_file)
            files_status['skipped_summarization'].append(media_file)
    
    return files_status


def process_parallel_pipeline(media_files, project_path, transcripts_path, summaries_path, 
                            project_name, whisper_model, openai_model, constants):
    """Process transcription and summarization in parallel."""
    # Import modules
    import transcribe_module
    import summarize_module
    
    # Analyze what needs to be processed
    file_status = analyze_file_status(media_files, transcripts_path, summaries_path, constants)
    
    print(f"File analysis:")
    print(f"  Need transcription: {len(file_status['need_transcription'])}")
    print(f"  Need summarization: {len(file_status['need_summarization'])}")
    print(f"  Fully processed: {len(file_status['fully_processed'])}")
    
    if file_status['skipped_transcription']:
        print(f"Skipping transcription for {len(file_status['skipped_transcription'])} files (already transcribed)")
    
    if file_status['skipped_summarization']:
        print(f"Skipping summarization for {len(file_status['skipped_summarization'])} files (already summarized)")
    
    if not file_status['need_transcription'] and not file_status['need_summarization']:
        print("All files are fully processed. No work to do.")
        return
    
    # Create queues for communication
    transcription_queue = queue.Queue()
    summary_queue = queue.Queue()
    
    # Add existing transcripts to the queue for summarization
    for media_file in file_status['need_summarization']:
        if media_file not in file_status['need_transcription']:
            # This file has transcript but needs summary
            transcript_path = transcribe_module.get_transcript_path(media_file, transcripts_path, constants)
            episode_number = media_files.index(media_file) + 1
            transcription_queue.put({
                'status': 'completed',
                'file': media_file,
                'transcript_path': transcript_path,
                'episode_number': episode_number
            })
    
    # Create worker threads
    transcription_thread = None
    if file_status['need_transcription']:
        transcription_thread = threading.Thread(
            target=transcribe_module.transcribe_worker,
            args=(file_status['need_transcription'], transcripts_path, whisper_model, constants, transcription_queue)
        )
    
    summarization_thread = None
    if file_status['need_summarization']:
        summarization_thread = threading.Thread(
            target=summarize_module.summarize_worker,
            args=(transcription_queue, project_path, transcripts_path, summaries_path, 
                  project_name, openai_model, constants, summary_queue)
        )
    
    # Start threads
    if transcription_thread:
        transcription_thread.start()
    else:
        # Signal that transcription is finished
        transcription_queue.put({'status': 'finished'})
    
    if summarization_thread:
        summarization_thread.start()
    
    # Monitor progress
    total_to_transcribe = len(file_status['need_transcription'])
    total_to_summarize = len(file_status['need_summarization'])
    transcribed = 0
    summarized = 0
    
    print(f"Processing {total_to_transcribe} transcriptions and {total_to_summarize} summaries...")
    
    # Wait for completion and show progress
    while (transcription_thread and transcription_thread.is_alive()) or (summarization_thread and summarization_thread.is_alive()):
        try:
            # Check for completed summaries
            while True:
                summary_result = summary_queue.get_nowait()
                if summary_result['status'] == 'completed':
                    summarized += 1
                    print(f"Progress: {transcribed}/{total_to_transcribe} transcribed, {summarized}/{total_to_summarize} summarized")
                elif summary_result['status'] == 'error':
                    print(f"Summary error: {summary_result['error']}")
                elif summary_result['status'] == 'finished':
                    break
        except queue.Empty:
            pass
        
        time.sleep(0.5)
    
    # Final cleanup
    if transcription_thread:
        transcription_thread.join()
    if summarization_thread:
        summarization_thread.join()
    
    print(f"Pipeline complete: {transcribed} transcribed, {summarized} summarized")


def main():
    """Main application entry point."""
    args = parse_arguments()
    
    print(f"VideoTranscribe Phase 2 - Processing project: {args.project_name}")
    print(f"Whisper model: {args.whisper_model}")
    if not args.transcribe_only:
        print(f"OpenAI model: {args.openai_model}")
    print("-" * 60)
    
    try:
        process_pipeline(args)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()