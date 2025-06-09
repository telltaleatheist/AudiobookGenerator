#!/usr/bin/env python3
"""
OpenAI Engine - OpenAI TTS processor with voice selection and quality control
UPDATED: Uses new section-based architecture with dynamic parameter loading and progress bar
"""

import sys
import os
import time
import requests # type: ignore
from pathlib import Path
from typing import List, Dict, Any

# Import dynamic utilities from engine registry
from core.progress_display_manager import log_status
from engines.base_engine import (
    extract_engine_config, 
    filter_params_for_function,
    create_generation_params,
    validate_required_params
)

# OpenAI imports
try:
    import openai # type: ignore
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("ERROR: OpenAI library not available. Install with: pip install openai")

def chunk_text_for_openai(text, max_chars=4000):
    """Split text into chunks optimized for OpenAI TTS"""
    # OpenAI TTS can handle very large chunks efficiently
    import re
    
    # If text is short enough, return as single chunk
    if len(text) <= max_chars:
        return [text]
    
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        
        # If paragraph alone exceeds max, split by sentences
        if len(paragraph) > max_chars:
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            for sentence in sentences:
                if current_chunk and len(current_chunk) + len(sentence) + 1 > max_chars:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
            continue
        
        # If adding this paragraph would exceed max, start new chunk
        if current_chunk and len(current_chunk) + len(paragraph) + 2 > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    
    # Add remaining text
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def validate_openai_config(openai_config):
    """Validate OpenAI configuration and check API key"""
    # Check API key
    api_key = openai_config.get('api_key') or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OpenAI API key not found")
        print("💡 Set OPENAI_API_KEY environment variable or add 'api_key' to config")
        return False
    
    # Validate model
    valid_models = ['tts-1', 'tts-1-hd', 'gpt-4o-mini-tts']
    model = openai_config.get('model', 'tts-1')
    if model not in valid_models:
        print(f"WARNING: Unknown model '{model}'. Valid models: {', '.join(valid_models)}")
    
    # Validate voice
    valid_voices = ['echo', 'fable', 'onyx', 'nova', 'shimmer', 'alloy']
    voice = openai_config.get('voice', 'onyx')
    if voice not in valid_voices:
        print(f"WARNING: Unknown voice '{voice}'. Valid voices: {', '.join(valid_voices)}")
    
    # Validate speed
    speed = openai_config.get('speed', 1.0)
    if speed is not None and not (0.25 <= speed <= 4.0):
        print(f"WARNING: Speed {speed} outside valid range (0.25-4.0)")
    
    return True

def setup_openai_client(openai_config):
    """Setup OpenAI client with API key"""
    api_key = openai_config.get('api_key') or os.getenv("OPENAI_API_KEY")
    
    if api_key:
        openai.api_key = api_key
        # For newer versions of openai library
        if hasattr(openai, 'OpenAI'):
            client = openai.OpenAI(api_key=api_key)
            return client
        else:
            return openai  # Older version, use module directly
    else:
        raise ValueError("OpenAI API key not found")

def generate_openai_audio_dynamic(client, text, openai_config):
    """Generate audio using OpenAI TTS with dynamic configuration support"""
    try:
        # Build generation parameters
        generation_params = {
            'model': openai_config.get('model', 'tts-1'),
            'voice': openai_config.get('voice', 'onyx'),
            'input': text
        }
        
        # Add optional parameters if present
        if 'speed' in openai_config and openai_config['speed'] is not None:
            generation_params['speed'] = openai_config['speed']
        
        if 'response_format' in openai_config and openai_config['response_format'] is not None:
            generation_params['response_format'] = openai_config['response_format']
        
        # Generate audio
        if hasattr(client, 'audio') and hasattr(client.audio, 'speech'):
            # Newer client interface
            response = client.audio.speech.create(**generation_params)
        else:
            # Older interface
            response = client.audio.speech.create(**generation_params)
        
        # Get the audio content
        if hasattr(response, 'content'):
            audio_content = response.content
        else:
            audio_content = response
        
        return audio_content
        
    except Exception as e:
        print(f"ERROR: OpenAI TTS generation failed: {e}")
        
        # Provide helpful error messages
        if "authentication" in str(e).lower():
            print("💡 Check your OpenAI API key")
        elif "quota" in str(e).lower():
            print("💡 Check your OpenAI account billing/quota")
        elif "rate" in str(e).lower():
            print("💡 Rate limited - wait a moment before retrying")
        
        return None

def save_openai_audio(audio_content, output_path, response_format='mp3'):
    """Save OpenAI audio content to file"""
    if audio_content is None:
        return False
    
    try:
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save based on format
        if response_format == 'mp3':
            # Save as MP3, then convert to WAV for consistency with other engines
            temp_mp3 = output_path.with_suffix('.mp3')
            with open(temp_mp3, 'wb') as f:
                f.write(audio_content)
            
            # Convert to WAV using ffmpeg if available
            try:
                import subprocess
                result = subprocess.run([
                    'ffmpeg', '-i', str(temp_mp3), '-y', str(output_path)
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    temp_mp3.unlink()  # Remove temp MP3
                else:
                    # If conversion fails, keep MP3
                    output_path = temp_mp3
            except FileNotFoundError:
                # ffmpeg not available, keep MP3
                output_path = temp_mp3
                
        else:
            # Save in original format
            with open(output_path, 'wb') as f:
                f.write(audio_content)
        
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to save OpenAI audio: {e}")
        return False

def estimate_openai_cost(text, model='tts-1'):
    """Estimate the cost for generating audio with OpenAI TTS"""
    char_count = len(text)
    
    # Pricing as of 2025 (per 1M characters)
    pricing = {
        'tts-1': 15.00,          # Standard TTS
        'tts-1-hd': 30.00,       # HD TTS
        'gpt-4o-mini-tts': 0.60  # Mini TTS (input cost)
    }
    
    cost_per_million = pricing.get(model, 15.00)
    estimated_cost = (char_count / 1_000_000) * cost_per_million
    
    return estimated_cost, char_count

def _update_openai_progress_bar(completed: int, total: int, chunk_times: list):
    """Update horizontal progress bar for OpenAI chunk processing"""
    # Calculate progress percentage
    if total == 0:
        percent = 100
    else:
        percent = (completed / total) * 100
    
    # Calculate ETA
    if completed == 0 or len(chunk_times) == 0:
        eta_str = "calculating..."
    elif completed >= total:
        eta_str = "complete!"
    else:
        # Use average of recent chunk times for ETA
        if len(chunk_times) >= 3:
            recent_times = chunk_times[-3:]
            avg_time = sum(recent_times) / len(recent_times)
        else:
            avg_time = sum(chunk_times) / len(chunk_times)
        
        remaining_chunks = total - completed
        remaining_seconds = remaining_chunks * avg_time
        
        # Format ETA
        if remaining_seconds < 60:
            eta_str = f"{int(remaining_seconds)}s"
        elif remaining_seconds < 3600:
            minutes = int(remaining_seconds // 60)
            seconds = int(remaining_seconds % 60)
            eta_str = f"{minutes}m {seconds}s"
        else:
            hours = int(remaining_seconds // 3600)
            minutes = int((remaining_seconds % 3600) // 60)
            eta_str = f"{hours}h {minutes}m"
    
    # Get terminal width (default to 80 if can't detect)
    try:
        import shutil
        terminal_width = shutil.get_terminal_size().columns
    except:
        terminal_width = 80
    
    # Build the components separately to ensure no string corruption
    prefix = "    🤖 OpenAI: "
    chunk_info = f"{completed}/{total} chunks"
    percent_info = f"({percent:.0f}%)"
    eta_info = f"ETA: {eta_str}"
    
    # Build suffix with proper spacing
    suffix = f" {chunk_info} {percent_info} {eta_info}"
    
    # Calculate available space for the bar with generous padding
    total_text_length = len(prefix) + len(suffix) + 2  # +2 for brackets []
    available_width = terminal_width - total_text_length - 5  # -5 for extra safety
    bar_width = max(5, min(30, available_width))  # Conservative bar width
    
    # Create progress bar
    if total > 0:
        filled_length = int(bar_width * completed // total)
    else:
        filled_length = bar_width
    bar = '█' * filled_length + '░' * (bar_width - filled_length)
    
    # Build complete line
    progress_line = f"{prefix}[{bar}]{suffix}"
    
    # Final safety check - if still too long, truncate the bar more
    while len(progress_line) > terminal_width - 2 and bar_width > 5:
        bar_width -= 1
        if total > 0:
            filled_length = int(bar_width * completed // total)
        else:
            filled_length = bar_width
        bar = '█' * filled_length + '░' * (bar_width - filled_length)
        progress_line = f"{prefix}[{bar}]{suffix}"
    
    # Clear the line first, then print the progress
    print(f"\r{' ' * (terminal_width - 1)}\r{progress_line}", end='', flush=True)

def process_openai_chunks_with_retry(client, chunks, output_dir, openai_config):
    """Process chunks with retry logic, cost estimation, and progress bar"""
    generated_files = []
    
    # Get configuration parameters
    retry_attempts = openai_config.get('retry_attempts', 3)
    retry_delay = openai_config.get('retry_delay', 1.0)
    model = openai_config.get('model', 'tts-1')
    response_format = openai_config.get('response_format', 'mp3')
    ignore_errors = openai_config.get('ignore_errors', False)
    skip_failed_chunks = openai_config.get('skip_failed_chunks', False)
    
    # Calculate total cost estimate
    total_chars = sum(len(chunk) for chunk in chunks)
    estimated_cost, _ = estimate_openai_cost(''.join(chunks), model)
    
    print(f"  📝 Processing {len(chunks)} chunks ({total_chars:,} characters)")
    print(f"  💰 Estimated cost: ${estimated_cost:.4f} USD")
    
    if estimated_cost > 1.0:  # Warn for costs over $1
        print(f"⚠️ WARNING: Estimated cost is ${estimated_cost:.2f}")
    
    chunk_times = []  # Track timing for ETA
    total_chunks = len(chunks)
    
    # Print initial progress bar
    _update_openai_progress_bar(0, total_chunks, chunk_times)

    for i, chunk_text in enumerate(chunks):
        chunk_num = i + 1
        chunk_start_time = time.time()
        
        # Use appropriate extension
        if response_format == 'mp3':
            output_file = output_dir / f"chunk_{chunk_num:03d}_openai.wav"  # Will convert to WAV
        else:
            output_file = output_dir / f"chunk_{chunk_num:03d}_openai.{response_format}"
        
        # Retry logic
        success = False
        for attempt in range(retry_attempts):
            try:
                # Generate audio
                audio_content = generate_openai_audio_dynamic(client, chunk_text, openai_config)
                
                if audio_content is None:
                    if attempt < retry_attempts - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        print(f"\n    ❌ All {retry_attempts} attempts failed for chunk {chunk_num}")
                        if not ignore_errors:
                            break
                        continue
                
                # Save audio
                if save_openai_audio(audio_content, output_file, response_format):
                    generated_files.append(str(output_file))
                    success = True
                    break
                else:
                    if attempt < retry_attempts - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        print(f"\n    ❌ Failed to save chunk {chunk_num} after {retry_attempts} attempts")
                
            except Exception as e:
                if attempt < retry_attempts - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"\n    ❌ Failed to process chunk {chunk_num} after {retry_attempts} attempts: {e}")
                    break
        
        if not success and not skip_failed_chunks:
            print(f"\n    ❌ Critical failure on chunk {chunk_num}")
            break
        
        # Record completion time and update progress bar
        chunk_duration = time.time() - chunk_start_time
        chunk_times.append(chunk_duration)
        _update_openai_progress_bar(chunk_num, total_chunks, chunk_times)
    
    # Clear progress bar and show completion
    print()  # New line after progress bar
    print("    ✅ All chunks processed")
    
    return generated_files

def process_openai_text_file(text_file: str, output_dir: str, config: Dict[str, Any], paths: Dict[str, Any]) -> List[str]:
    """Main OpenAI engine processor with new architecture"""
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI library not available. Install with: pip install openai")
    
    try:
        # Extract ALL OpenAI config parameters dynamically
        openai_config = extract_engine_config(config, 'openai', verbose=True)
        
        # Validate required parameters
        required_params = ['voice', 'model', 'retry_attempts', 'retry_delay', 'ignore_errors', 'skip_failed_chunks']
        missing_params = validate_required_params(openai_config, required_params, 'openai')
        if missing_params:
            print(f"ERROR: Missing required OpenAI configuration: {', '.join(missing_params)}")
            return []
        
        log_status(f"Starting OpenAI TTS processing")
        log_status(f"Model: {openai_config['model']}")
        log_status(f"Voice: {openai_config['voice']}")
        
        # Display configured parameters
        active_params = {k: v for k, v in openai_config.items() if v is not None and k != 'api_key'}
        if openai_config.get('verbose', False):
            log_status(f"Active parameters: {active_params}")
        
        # Validate configuration
        if not validate_openai_config(openai_config):
            return []
        
        # Setup OpenAI client
        try:
            client = setup_openai_client(openai_config)
            log_status(f"OpenAI client initialized successfully")
        except Exception as e:
            print(f"ERROR: Failed to initialize OpenAI client: {e}")
            return []
        
        # Read clean text (already preprocessed by pipeline)
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        if not text:
            print(f"ERROR: No text content to process")
            return []
        
        # Chunk text for OpenAI TTS
        chunk_max_chars = openai_config.get('chunk_max_chars', 4000)
        chunks = chunk_text_for_openai(text, chunk_max_chars)
        log_status(f"Created {len(chunks)} chunks for OpenAI TTS")
        
        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process chunks with retry logic and cost tracking
        generated_files = process_openai_chunks_with_retry(client, chunks, output_dir, openai_config)
        
        # Final statistics
        success_rate = len(generated_files) / len(chunks) * 100 if chunks else 0
        log_status(f"OpenAI TTS processing completed: {len(generated_files)}/{len(chunks)} files generated ({success_rate:.1f}% success)")
        
        if len(generated_files) == 0:
            print(f"ERROR: No audio files were generated successfully")
        
        return generated_files
        
    except Exception as e:
        print(f"ERROR: OpenAI TTS processing failed: {e}")
        return []

def register_openai_engine():
    """Register OpenAI engine with the registry"""
    from engines.base_engine import register_engine
    register_engine('openai', process_openai_text_file)
