"""
VideoTranscribe - Transcription Module
Contains all transcription-related functionality using OpenAI Whisper.

Copyright (C) 2025 Martin 'quill18' Glaude

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
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



def transcribe_worker(files_to_transcribe, all_media_files, transcripts_path, whisper_model, constants, transcription_queue):
    """Worker function for transcription processing."""
    if not files_to_transcribe:
        print("All files have been transcribed. No work to do.")
        # Signal completion
        transcription_queue.put({'status': 'finished'})
        return
    
    print(f"Transcription worker: Processing {len(files_to_transcribe)} files...")
    
    # Load Whisper model
    try:
        model = load_whisper_model(whisper_model)
    except Exception as e:
        transcription_queue.put({'status': 'error', 'error': f"Failed to load model: {e}"})
        return
    
    # Sort all media files to establish consistent episode order
    all_media_files_sorted = sorted(all_media_files, key=lambda x: x.name.lower())
    
    # Process files
    for media_file in files_to_transcribe:
        # Get correct episode number based on position in full sorted list of ALL media files
        episode_number = all_media_files_sorted.index(media_file) + 1
        
        print(f"\nTranscribing episode {episode_number}: {media_file.name}")
        transcript_path = get_transcript_path(media_file, transcripts_path, constants)
        
        if transcribe_file(model, media_file, transcript_path, constants):
            # Notify summarization queue that transcript is ready
            transcription_queue.put({
                'status': 'completed',
                'file': media_file,
                'transcript_path': transcript_path,
                'episode_number': episode_number
            })
        else:
            transcription_queue.put({
                'status': 'error',
                'file': media_file,
                'error': f"Failed to transcribe {media_file.name}"
            })
    
    # Signal that all transcription is complete
    transcription_queue.put({'status': 'finished'})