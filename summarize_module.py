"""
VideoTranscribe - Summarization Module
Contains all summarization-related functionality using OpenAI API for Let's Play videos.
"""

import os
import sys
import time
import queue
from pathlib import Path
from openai import OpenAI


def get_summary_path(media_file, summaries_path, constants):
    """Get the summary file path for a given media file."""
    summary_name = media_file.stem + constants['SUMMARY_EXTENSION']
    return summaries_path / summary_name


def summary_exists(media_file, summaries_path, constants):
    """Check if summary already exists for the media file."""
    summary_path = get_summary_path(media_file, summaries_path, constants)
    return summary_path.exists()


def extract_game_name_from_project(project_name):
    """Extract game name from project name (basic heuristic)."""
    # Remove common prefixes/suffixes and replace underscores with spaces
    game_name = project_name.replace("_", " ").replace("-", " ")
    
    # Remove common Let's Play prefixes
    prefixes_to_remove = ["lets play", "lp", "playthrough", "gameplay"]
    for prefix in prefixes_to_remove:
        if game_name.lower().startswith(prefix):
            game_name = game_name[len(prefix):].strip()
    
    return game_name.title()


def read_transcript(transcript_path, constants):
    """Read transcript from file."""
    try:
        with open(transcript_path, 'r', encoding=constants['ENCODING']) as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading transcript {transcript_path}: {e}")
        return None


def read_previous_summaries(summaries_path, current_episode_number, constants):
    """Read all previous episode summaries to provide context."""
    previous_summaries = []
    
    # Get all summary files and sort them
    summary_files = []
    if summaries_path.exists():
        for summary_file in summaries_path.iterdir():
            if summary_file.suffix == constants['SUMMARY_EXTENSION'].split('.')[1]:  # .txt
                summary_files.append(summary_file)
    
    # Sort by filename to get proper episode order
    summary_files.sort(key=lambda x: x.name.lower())
    
    # Read summaries for episodes before the current one
    for summary_file in summary_files:
        try:
            with open(summary_file, 'r', encoding=constants['ENCODING']) as f:
                content = f.read().strip()
                if content:
                    previous_summaries.append(f"Episode: {summary_file.stem}\n{content}")
        except Exception as e:
            print(f"Warning: Could not read previous summary {summary_file}: {e}")
    
    return previous_summaries


def create_openai_client():
    """Create OpenAI client with API key from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    return OpenAI(api_key=api_key)


def generate_summary(transcript, game_name, project_name, episode_number, previous_summaries, openai_model, constants):
    """Generate Let's Play summary using OpenAI API."""
    try:
        client = create_openai_client()
        
        # Prepare previous context
        if previous_summaries:
            previous_context = constants['CONTEXT_TEMPLATE'].format(
                previous_summaries="\n\n".join(previous_summaries[-3:])  # Last 3 episodes for context
            )
        else:
            previous_context = "This is the first episode of the Let's Play series.\n\n---"
        
        # Create the prompt
        prompt = constants['LETS_PLAY_PROMPT_TEMPLATE'].format(
            game_name=game_name,
            episode_number=episode_number,
            project_name=project_name,
            previous_context=previous_context,
            transcript=transcript
        )
        
        # Debug logging
        if constants.get('DEBUG', False):
            print("\n" + "="*80)
            print("DEBUG: OpenAI API Request")
            print("="*80)
            print(f"Model: {openai_model}")
            print(f"Max Tokens: {constants['OPENAI_MAX_TOKENS']}")
            print(f"Temperature: {constants['OPENAI_TEMPERATURE']}")
            print("\nSystem Prompt:")
            print("-" * 40)
            print(constants['LETS_PLAY_SYSTEM_PROMPT'])
            print("\nUser Prompt:")
            print("-" * 40)
            print(prompt)
            print("="*80)
        
        # Make API call
        response = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": constants['LETS_PLAY_SYSTEM_PROMPT']},
                {"role": "user", "content": prompt}
            ],
            max_tokens=constants['OPENAI_MAX_TOKENS'],
            temperature=constants['OPENAI_TEMPERATURE']
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error generating summary: {e}")
        return None


def save_summary(summary_content, summary_path, constants):
    """Save summary to file."""
    try:
        with open(summary_path, 'w', encoding=constants['ENCODING']) as f:
            f.write(summary_content)
        return True
    except Exception as e:
        print(f"Error saving summary to {summary_path}: {e}")
        return False


def summarize_transcript(media_file, transcript_path, summaries_path, project_name, episode_number, openai_model, constants):
    """Summarize a single transcript file."""
    # Check if summary already exists
    summary_path = get_summary_path(media_file, summaries_path, constants)
    if summary_path.exists():
        print(f"Summary already exists: {summary_path.name}")
        return True
    
    # Read transcript
    transcript = read_transcript(transcript_path, constants)
    if not transcript:
        print(f"Could not read transcript for {media_file.name}")
        return False
    
    print(f"Generating summary for: {media_file.name}")
    
    # Extract game name
    game_name = extract_game_name_from_project(project_name)
    
    # Read previous summaries for context
    previous_summaries = read_previous_summaries(summaries_path, episode_number, constants)
    
    # Generate summary
    summary = generate_summary(
        transcript, game_name, project_name, episode_number, 
        previous_summaries, openai_model, constants
    )
    
    if not summary:
        print(f"Failed to generate summary for {media_file.name}")
        return False
    
    # Save summary
    if save_summary(summary, summary_path, constants):
        print(f"Summary saved: {summary_path.name}")
        return True
    else:
        return False


def summarize_existing_transcripts(project_path, transcripts_path, summaries_path, project_name, openai_model, constants):
    """Summarize all existing transcripts (synchronous version)."""
    # Find all transcript files
    transcript_files = []
    if transcripts_path.exists():
        for transcript_file in transcripts_path.iterdir():
            if transcript_file.suffix == constants['TRANSCRIPT_EXTENSION']:
                transcript_files.append(transcript_file)
    
    if not transcript_files:
        print("No transcript files found to summarize.")
        return
    
    # Sort files to process in order
    transcript_files.sort(key=lambda x: x.name.lower())
    
    # Find corresponding media files for episode numbering
    media_files = []
    for transcript_file in transcript_files:
        # Find corresponding media file by stem name
        media_stem = transcript_file.stem
        for file_path in project_path.iterdir():
            if file_path.is_file() and file_path.stem == media_stem:
                if file_path.suffix.lower() in constants['SUPPORTED_EXTENSIONS']:
                    media_files.append(file_path)
                    break
        else:
            # Create a dummy media file object for processing
            media_files.append(transcript_file.with_suffix('.mp4'))  # Dummy extension
    
    # Check which summaries need to be created
    files_to_process = []
    skipped_files = []
    
    for i, (transcript_file, media_file) in enumerate(zip(transcript_files, media_files), 1):
        if summary_exists(media_file, summaries_path, constants):
            skipped_files.append(media_file)
        else:
            files_to_process.append((transcript_file, media_file, i))
    
    if skipped_files:
        print(f"Skipping {len(skipped_files)} files (summaries already exist):")
        for skipped_file in skipped_files:
            print(f"  - {skipped_file.stem}")
    
    if not files_to_process:
        print("All transcripts have been summarized. No work to do.")
        return
    
    print(f"Summarizing {len(files_to_process)} transcripts...")
    print("-" * 50)
    
    # Process files
    successful = 0
    failed = 0
    
    for transcript_file, media_file, episode_number in files_to_process:
        print(f"\nSummarizing episode {episode_number}: {media_file.stem}")
        
        if summarize_transcript(media_file, transcript_file, summaries_path, project_name, episode_number, openai_model, constants):
            successful += 1
        else:
            failed += 1
        
        # Brief pause between API calls to be respectful
        time.sleep(1)
    
    # Summary
    print("-" * 50)
    print(f"Summarization complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {len(skipped_files)}")


def summarize_worker(transcription_queue, project_path, transcripts_path, summaries_path, project_name, openai_model, constants, summary_queue):
    """Worker function for threaded summarization processing."""
    print("Summarization worker: Starting...")
    
    processed_count = 0
    
    while True:
        try:
            # Wait for transcription to complete
            item = transcription_queue.get(timeout=5.0)
            
            if item['status'] == 'finished':
                print("Summarization worker: All transcriptions completed")
                break
            elif item['status'] == 'completed' and item['file'] is not None:
                # Process this completed transcription
                media_file = item['file']
                transcript_path = item['transcript_path']
                episode_number = item['episode_number']
                
                print(f"Summarization worker: Processing {media_file.name}")
                
                if summarize_transcript(media_file, transcript_path, summaries_path, project_name, episode_number, openai_model, constants):
                    processed_count += 1
                    summary_queue.put({'status': 'completed', 'file': media_file})
                    print(f"Summarization worker: Completed {media_file.name}")
                else:
                    summary_queue.put({'status': 'error', 'file': media_file, 'error': f"Failed to summarize {media_file.name}"})
                
                # Brief pause between API calls
                time.sleep(1)
                
            elif item['status'] == 'error':
                print(f"Summarization worker: Transcription error - {item.get('error', 'Unknown error')}")
                summary_queue.put({'status': 'error', 'error': item.get('error', 'Transcription failed')})
        
        except queue.Empty:
            # Timeout waiting for transcription - check if we should continue
            continue
        except Exception as e:
            print(f"Summarization worker error: {e}")
            summary_queue.put({'status': 'error', 'error': str(e)})
            break
    
    print(f"Summarization worker: Completed processing {processed_count} files")
    summary_queue.put({'status': 'finished', 'processed_count': processed_count})