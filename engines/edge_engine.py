#!/usr/bin/env python3
"""
Edge Engine - Free EdgeTTS processor (command-line version)
UPDATED: Uses new section-based architecture with dynamic parameter loading and progress bar
"""

import sys
import re
import time
import asyncio
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

# EdgeTTS imports (free version)
try:
    import edge_tts
    EDGE_AVAILABLE = True
except ImportError:
    EDGE_AVAILABLE = False
    print("ERROR: EdgeTTS not available. Install with: pip install edge-tts", file=sys.stderr)

def chunk_text_for_edge(text, max_chars=1000):
    """Split text into chunks optimized for EdgeTTS (free version can handle large chunks)"""
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
        
        # If we're at a good size, create chunk
        if len(current_chunk) >= max_chars * 0.8:  # 80% of max
            chunks.append(current_chunk.strip())
            current_chunk = ""
    
    # Add remaining text
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def apply_edge_text_preprocessing(text, edge_config):
    """Apply EdgeTTS text preprocessing based on config"""
    if not edge_config.get('normalize_text', True):
        return text
    
    processed_text = text
    
    # Apply various preprocessing based on config
    if edge_config.get('expand_abbreviations', True):
        # Basic abbreviation expansion for better TTS
        abbreviations = {
            'Dr.': 'Doctor',
            'Mr.': 'Mister', 
            'Mrs.': 'Missus',
            'Ms.': 'Miss',
            'Prof.': 'Professor',
            'etc.': 'and so on',
            'e.g.': 'for example',
            'i.e.': 'that is',
            'vs.': 'versus',
            'Inc.': 'Incorporated',
            'Ltd.': 'Limited',
            'Corp.': 'Corporation',
        }
        for abbrev, expansion in abbreviations.items():
            processed_text = re.sub(r'\b' + re.escape(abbrev) + r'\b', expansion, processed_text, flags=re.IGNORECASE)
    
    # Number handling
    if edge_config.get('spell_out_numbers', False):
        # Simple single digit spelling (basic implementation)
        number_words = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
        }
        for digit, word in number_words.items():
            processed_text = re.sub(r'\b' + digit + r'\b', word, processed_text)
    
    # Clean up extra whitespace
    processed_text = re.sub(r'\s+', ' ', processed_text)
    
    return processed_text.strip()

async def generate_edge_audio_dynamic(text, edge_config):
    """Generate audio using free EdgeTTS with dynamic configuration"""
    try:
        # Validate text
        if not text or not text.strip():
            print(f"ERROR: Empty text provided to EdgeTTS", file=sys.stderr)
            return None
        
        # Apply text preprocessing
        processed_text = apply_edge_text_preprocessing(text.strip(), edge_config)
        
        # Build EdgeTTS communicate parameters (only free version parameters)
        communicate_params = {
            'text': processed_text,
            'voice': edge_config.get('voice', 'en-US-AriaNeural')
        }
        
        # Add free EdgeTTS parameters (rate, pitch, volume only)
        free_edge_params = ['rate', 'pitch', 'volume']
        for param in free_edge_params:
            if param in edge_config and edge_config[param] is not None:
                communicate_params[param] = edge_config[param]
        
        # Create communicate object with filtered parameters
        valid_params = filter_params_for_function(communicate_params, edge_tts.Communicate.__init__, verbose=edge_config.get('debug_output', False))
        
        # Remove 'self' parameter that's not needed for instantiation
        if 'self' in valid_params:
            del valid_params['self']
        
        communicate = edge_tts.Communicate(**valid_params)
        
        # Collect audio data
        audio_data = b""
        chunk_count = 0
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
                chunk_count += 1
        
        if len(audio_data) == 0:
            print(f"ERROR: No audio data received from EdgeTTS", file=sys.stderr)
            return None
        
        return audio_data
        
    except Exception as e:
        print(f"ERROR: EdgeTTS generation failed: {e}", file=sys.stderr)
        if edge_config.get('debug_output', False):
            import traceback
            traceback.print_exc(file=sys.stderr)
        return None

def save_edge_audio_dynamic(audio_data, output_path, edge_config):
    """Save EdgeTTS audio data with optional post-processing"""
    if not audio_data:
        print(f"ERROR: No audio data to save", file=sys.stderr)
        return False
        
    try:
        # Note: Free EdgeTTS returns MP3 data, so post-processing is limited
        # Most audio effects would require conversion to WAV first
        
        # Save the audio (EdgeTTS free version outputs MP3 by default)
        output_path = Path(output_path)
        
        # Save as MP3 first, then convert to WAV if ffmpeg available
        temp_mp3 = output_path.with_suffix('.mp3')
        with open(temp_mp3, 'wb') as f:
            f.write(audio_data)
        
        # Convert to WAV using ffmpeg if available
        try:
            import subprocess
            result = subprocess.run([
                'ffmpeg', '-i', str(temp_mp3), '-y', str(output_path)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                temp_mp3.unlink()  # Remove temp MP3
            else:
                # If conversion fails, rename MP3 to final name
                temp_mp3.rename(output_path.with_suffix('.mp3'))
        except FileNotFoundError:
            # ffmpeg not available, rename MP3 to final name
            temp_mp3.rename(output_path.with_suffix('.mp3'))
        
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to save audio to {output_path}: {e}", file=sys.stderr)
        return False

def _update_edge_progress_bar(completed: int, total: int, chunk_times: list):
    """Update horizontal progress bar for Edge chunk processing"""
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
    prefix = "    🌐 Edge: "
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

async def process_edge_chunks_async_dynamic(chunks, output_dir, edge_config):
    """Process chunks with EdgeTTS asynchronously with dynamic retry logic and progress bar"""
    generated_files = []
    
    # Get retry settings from config
    retry_attempts = edge_config.get('retry_attempts', 3)
    retry_delay = edge_config.get('retry_delay', 2.0)
    delay_between_chunks = edge_config.get('delay', 1.5)
    ignore_errors = edge_config.get('ignore_errors', False)
    skip_failed_chunks = edge_config.get('skip_failed_chunks', False)
    
    chunk_times = []  # Track timing for ETA
    total_chunks = len(chunks)
    
    print(f"  📝 Processing {len(chunks)} chunks with EdgeTTS...")
    
    # Print initial progress bar
    _update_edge_progress_bar(0, total_chunks, chunk_times)
    
    for i, chunk_text in enumerate(chunks):
        chunk_num = i + 1
        chunk_start_time = time.time()
        output_file = output_dir / f"chunk_{chunk_num:03d}_edge.wav"
        
        # Retry logic
        success = False
        for attempt in range(retry_attempts):
            try:
                # Generate audio with dynamic config
                audio_data = await generate_edge_audio_dynamic(chunk_text, edge_config)
                
                if audio_data is None:
                    if attempt < retry_attempts - 1:
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        print(f"\n    ❌ All {retry_attempts} attempts failed for chunk {chunk_num}")
                        if not ignore_errors:
                            break
                        continue
                
                # Save audio with dynamic config
                if save_edge_audio_dynamic(audio_data, output_file, edge_config):
                    generated_files.append(str(output_file))
                    success = True
                    break
                else:
                    if attempt < retry_attempts - 1:
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        print(f"\n    ❌ Failed to save chunk {chunk_num} after {retry_attempts} attempts")
                
            except Exception as e:
                if attempt < retry_attempts - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    print(f"\n    ❌ Failed to process chunk {chunk_num} after {retry_attempts} attempts: {e}")
                    if edge_config.get('debug_output', False):
                        import traceback
                        traceback.print_exc(file=sys.stderr)
                    break
        
        if not success and not skip_failed_chunks:
            print(f"\n    ❌ Critical failure on chunk {chunk_num}")
            break
        
        # Record completion time and update progress bar
        chunk_duration = time.time() - chunk_start_time
        chunk_times.append(chunk_duration)
        _update_edge_progress_bar(chunk_num, total_chunks, chunk_times)
        
        # Add delay between chunks to avoid throttling (except for last chunk)
        if i < len(chunks) - 1 and delay_between_chunks > 0:
            await asyncio.sleep(delay_between_chunks)
    
    # Clear progress bar and show completion
    print()  # New line after progress bar
    print("    ✅ All chunks processed")
    
    return generated_files

def test_edge_voice_dynamic(voice_name, edge_config):
    """Test if a voice is available in free EdgeTTS with enhanced error handling"""
    async def _test():
        try:
            voices = await edge_tts.list_voices()
            available_voices = [v["Name"] for v in voices]
            
            if edge_config.get('verbose', False):
                log_status(f"Found {len(available_voices)} total voices", file=sys.stderr)
            
            # Check if the requested voice exists
            voice_exists = voice_name in available_voices
            
            if edge_config.get('verbose', False):
                log_status(f"Voice '{voice_name}' exists: {voice_exists}", file=sys.stderr)
            
            # Get language-specific voices for fallback
            lang_prefix = voice_name.split('-')[0:2]  # e.g., ['en', 'US']
            if len(lang_prefix) >= 2:
                lang_pattern = f"{lang_prefix[0]}-{lang_prefix[1]}"
                lang_voices = [v for v in available_voices if v.startswith(lang_pattern)]
            else:
                lang_voices = [v for v in available_voices if v.startswith('en-')]
            
            if edge_config.get('verbose', False):
                log_status(f"Found {len(lang_voices)} voices for language pattern", file=sys.stderr)
            
            return voice_exists, lang_voices, available_voices
            
        except Exception as e:
            print(f"ERROR: Could not list EdgeTTS voices: {e}", file=sys.stderr)
            if edge_config.get('debug_output', False):
                import traceback
                traceback.print_exc(file=sys.stderr)
            return False, [], []
    
    return asyncio.run(_test())

def process_edge_text_file(text_file: str, output_dir: str, config: Dict[str, Any], paths: Dict[str, Any]) -> List[str]:
    """Main EdgeTTS engine processor with new architecture (FREE VERSION ONLY)"""
    if not EDGE_AVAILABLE:
        raise ImportError("EdgeTTS not available. Install with: pip install edge-tts")
    
    try:
        # Extract ALL EdgeTTS config parameters dynamically
        edge_config = extract_engine_config(config, 'edge', verbose=True)
        
        # Validate required parameters
        required_params = ['voice', 'rate', 'pitch', 'volume', 'chunk_max_chars', 'delay', 
                          'normalize_text', 'expand_abbreviations', 'retry_attempts', 'retry_delay', 
                          'ignore_errors', 'skip_failed_chunks', 'verbose', 'debug_output']
        missing_params = validate_required_params(edge_config, required_params, 'edge')
        if missing_params:
            print(f"ERROR: Missing required EdgeTTS configuration: {', '.join(missing_params)}", file=sys.stderr)
            return []
        
        log_status(f"Starting EdgeTTS processing (free version)", file=sys.stderr)
        log_status(f"Voice: {edge_config['voice']}", file=sys.stderr)
        
        # Show free EdgeTTS parameters
        rate = edge_config.get('rate', '+0%')
        pitch = edge_config.get('pitch', '+0Hz')
        volume = edge_config.get('volume', '+0%')
        log_status(f"Rate: {rate}, Pitch: {pitch}, Volume: {volume}", file=sys.stderr)
        
        # Display configured features (only those supported by free version)
        free_features = []
        if edge_config.get('streaming'):
            free_features.append("streaming")
        if edge_config.get('normalize_text'):
            free_features.append("text normalization")
        if edge_config.get('expand_abbreviations'):
            free_features.append("abbreviation expansion")
        if edge_config.get('retry_attempts', 1) > 1:
            free_features.append(f"retry x{edge_config['retry_attempts']}")
        
        if free_features:
            log_status(f"Free EdgeTTS features: {', '.join(free_features)}", file=sys.stderr)
        
        # Test voice availability with enhanced checking
        voice_name = edge_config['voice']
        log_status(f"Validating voice availability", file=sys.stderr)
        voice_available, lang_voices, all_voices = test_edge_voice_dynamic(voice_name, edge_config)
        
        if not voice_available:
            print(f"WARNING: Voice {voice_name} not found", file=sys.stderr)
            
            # Check for fallback voice
            fallback_voice = edge_config.get('fallback_voice')
            if fallback_voice and fallback_voice in all_voices:
                log_status(f"Using fallback voice: {fallback_voice}", file=sys.stderr)
                edge_config['voice'] = fallback_voice
            elif lang_voices:
                log_status(f"Available voices for language: {lang_voices[:3]}", file=sys.stderr)
                log_status(f"Proceeding with {voice_name} anyway", file=sys.stderr)
            else:
                print(f"ERROR: Could not find suitable voice, but proceeding anyway", file=sys.stderr)
        else:
            log_status(f"Voice {voice_name} confirmed available", file=sys.stderr)
        
        # Read clean text (already processed in preprocessing)
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            if edge_config.get('verbose', False):
                log_status(f"Read {len(text)} characters from {text_file}", file=sys.stderr)
            
            if not text:
                print(f"ERROR: Text file is empty", file=sys.stderr)
                return []
                
        except Exception as e:
            print(f"ERROR: Could not read text file {text_file}: {e}", file=sys.stderr)
            return []
        
        # Chunk text for EdgeTTS
        chunk_max_chars = edge_config.get('chunk_max_chars', 1000)
        chunks = chunk_text_for_edge(text, chunk_max_chars)
        log_status(f"Created {len(chunks)} chunks for EdgeTTS", file=sys.stderr)
        
        if not chunks:
            print(f"ERROR: No chunks created from text", file=sys.stderr)
            return []
        
        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process chunks asynchronously with dynamic features
        try:
            generated_files = asyncio.run(process_edge_chunks_async_dynamic(chunks, output_dir, edge_config))
        except Exception as e:
            print(f"ERROR: Async processing failed: {e}", file=sys.stderr)
            if edge_config.get('debug_output', False):
                import traceback
                traceback.print_exc(file=sys.stderr)
            return []
        
        # Final statistics
        success_rate = len(generated_files) / len(chunks) * 100 if chunks else 0
        log_status(f"EdgeTTS processing completed: {len(generated_files)}/{len(chunks)} files generated ({success_rate:.1f}% success)", file=sys.stderr)
        
        if len(generated_files) == 0:
            print(f"ERROR: No audio files were generated successfully", file=sys.stderr)
        
        return generated_files
        
    except Exception as e:
        print(f"ERROR: EdgeTTS processing failed: {e}", file=sys.stderr)
        return []

def register_edge_engine():
    """Register EdgeTTS (free) engine with the registry"""
    from engines.base_engine import register_engine
    register_engine('edge', process_edge_text_file)