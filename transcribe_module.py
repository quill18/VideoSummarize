"""
VideoTranscribe - Transcription Module
Contains all transcription-related functionality using OpenAI Whisper.
"""

import sys
import whisper
import queue


def get_transcript_path(media_file, transcripts_path, constants):
    """Get the transcript file path for a given media file."""
    transcript_name = media_file.stem + constants['TRANSCRIPT_EXTENSION']
    return transcripts_path / transcript_name


def transcript_exists(media_file, transcripts_path, constants):
    """Check if transcript already exists for the media file."""
    transcript_path = get_transcript_path(media_file, transcripts_path, constants)
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


def transcribe_file(model, media_file, transcript_path, constants):
    """Transcribe a single media file."""
    try:
        print(f"Processing: {media_file.name}")
        
        # Transcribe with progress bar
        result = model.transcribe(str(media_file), verbose=False)
        
        # Save transcript
        with open(transcript_path, 'w', encoding=constants['ENCODING']) as f:
            f.write(result["text"])
        
        print(f"Transcript saved: {transcript_path.name}")
        return True
        
    except Exception as e:
        print(f"Error transcribing {media_file.name}: {e}")
        return False


def transcribe_media_files(media_files, transcripts_path, whisper_model, constants):
    """Transcribe a list of media files (synchronous version)."""
    # Check for existing transcripts
    files_to_process = []
    skipped_files = []
    
    for media_file in media_files:
        if transcript_exists(media_file, transcripts_path, constants):
            skipped_files.append(media_file)
        else:
            files_to_process.append(media_file)
    
    if skipped_files:
        print(f"Skipping {len(skipped_files)} files (transcripts already exist):")
        for skipped_file in skipped_files:
            print(f"  - {skipped_file.name}")
    
    if not files_to_process:
        print("All files have been transcribed. No work to do.")
        return
    
    print(f"Processing {len(files_to_process)} files...")
    print("-" * 50)
    
    # Load Whisper model
    model = load_whisper_model(whisper_model)
    
    # Process files
    successful = 0
    failed = 0
    
    for i, media_file in enumerate(files_to_process, 1):
        print(f"\nFile {i} of {len(files_to_process)}")
        transcript_path = get_transcript_path(media_file, transcripts_path, constants)
        
        if transcribe_file(model, media_file, transcript_path, constants):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print("-" * 50)
    print(f"Transcription complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {len(skipped_files)}")


def transcribe_worker(media_files, transcripts_path, whisper_model, constants, transcription_queue):
    """Worker function for threaded transcription processing."""
    # Check for existing transcripts
    files_to_process = []
    skipped_files = []
    
    for media_file in media_files:
        if transcript_exists(media_file, transcripts_path, constants):
            skipped_files.append(media_file)
        else:
            files_to_process.append(media_file)
    
    if skipped_files:
        print(f"Skipping {len(skipped_files)} files (transcripts already exist):")
        for skipped_file in skipped_files:
            print(f"  - {skipped_file.name}")
    
    if not files_to_process:
        print("All files have been transcribed. No work to do.")
        # Signal completion
        transcription_queue.put({'status': 'completed', 'file': None})
        return
    
    print(f"Transcription worker: Processing {len(files_to_process)} files...")
    
    # Load Whisper model
    try:
        model = load_whisper_model(whisper_model)
    except Exception as e:
        transcription_queue.put({'status': 'error', 'error': f"Failed to load model: {e}"})
        return
    
    # Process files
    for i, media_file in enumerate(files_to_process, 1):
        print(f"\nTranscribing file {i} of {len(files_to_process)}: {media_file.name}")
        transcript_path = get_transcript_path(media_file, transcripts_path, constants)
        
        if transcribe_file(model, media_file, transcript_path, constants):
            # Notify summarization queue that transcript is ready
            transcription_queue.put({
                'status': 'completed',
                'file': media_file,
                'transcript_path': transcript_path,
                'episode_number': i
            })
        else:
            transcription_queue.put({
                'status': 'error',
                'file': media_file,
                'error': f"Failed to transcribe {media_file.name}"
            })
    
    # Signal that all transcription is complete
    transcription_queue.put({'status': 'finished'})