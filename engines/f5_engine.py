#!/usr/bin/env python3
"""
F5 Engine - F5-TTS processor with natural speech synthesis
UPDATED: Uses new section-based architecture with dynamic parameter loading and progress bar
"""

import sys
import time
from core.progress_display_manager import log_status
import torch # type: ignore
from pathlib import Path
from typing import List, Dict, Any

# Import dynamic utilities from engine registry
from engines.base_engine import (
    extract_engine_config, 
    filter_params_for_function,
    create_generation_params,
    validate_required_params
)

# F5-TTS imports
try:
    from f5_tts.api import F5TTS # type: ignore
    import torchaudio # type: ignore
    F5_AVAILABLE = True
except ImportError:
    F5_AVAILABLE = False
    print("ERROR: F5-TTS not available. Install with: pip install f5-tts")

def find_companion_text_file(audio_file_path):
    """Find companion text file for the given audio file"""
    audio_path = Path(audio_file_path)
    
    # Look for text file with same name
    companion_text_path = audio_path.with_suffix('.txt')
    
    if companion_text_path.exists():
        try:
            with open(companion_text_path, 'r', encoding='utf-8') as f:
                companion_text = f.read().strip()
            
            if companion_text:
                log_status(f"Found companion text file: {companion_text_path.name}")
                log_status(f"Companion text ({len(companion_text)} chars): {companion_text[:100]}{'...' if len(companion_text) > 100 else ''}")
                return companion_text
            else:
                print(f"WARNING: Companion text file is empty: {companion_text_path.name}")
                return ""
        except Exception as e:
            print(f"WARNING: Could not read companion text file {companion_text_path.name}: {e}")
            return ""
    
    log_status(f"No companion text file found for {audio_path.name}")
    return ""

def auto_detect_reference_audio_and_text(project_paths):
    """Auto-detect reference audio and companion text from samples directory"""
    ref_audio = None
    ref_text = ""
    
    # Look for samples directory in project
    if 'project_dir' in project_paths:
        samples_dir = Path(project_paths['project_dir']) / 'samples'
    else:
        # Fallback: look for samples relative to other paths
        for path_key, path_value in project_paths.items():
            if isinstance(path_value, (str, Path)):
                potential_samples = Path(path_value).parent.parent / 'samples'
                if potential_samples.exists():
                    samples_dir = potential_samples
                    break
        else:
            print(f"WARNING: Could not locate samples directory")
            return None, ""
    
    if not samples_dir.exists():
        log_status(f"No samples directory found at {samples_dir}")
        return None, ""
    
    # Find first .wav file in samples directory
    wav_files = list(samples_dir.glob("*.wav"))
    if not wav_files:
        log_status(f"No .wav files found in {samples_dir}")
        return None, ""
    
    # Use first wav file found
    ref_audio = str(wav_files[0])
    log_status(f"Auto-detected reference audio: {wav_files[0].name}")
    
    # Look for companion text file
    ref_text = find_companion_text_file(ref_audio)
    
    return ref_audio, ref_text

def load_f5_model(f5_config):
    """Load F5-TTS model with configuration"""
    log_status(f"Loading F5 model")
    
    try:
        # F5TTS() typically doesn't take model parameters in __init__
        # The model is specified during inference, not initialization
        log_status(f"Initializing F5TTS API (model will be set during inference)")
        
        # Initialize F5TTS API with no parameters - model specified during inference
        f5tts = F5TTS()
        
        print("STATUS: F5 model loaded successfully")
        return f5tts
        
    except Exception as e:
        print(f"ERROR: Failed to load F5 model: {e}")
        return None

def chunk_text_for_f5(text, f5_config):
    """Chunk text for F5-TTS based on configuration"""
    max_chars = f5_config.get('chunk_max_chars', 350)
    
    # F5-TTS can handle longer chunks, but we still chunk for memory management
    if len(text) <= max_chars:
        return [text]
    
    import re
    
    # Split by paragraphs first
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

def generate_f5_audio_dynamic(f5tts, text, f5_config):
    """Generate audio using F5-TTS with dynamic configuration support"""
    try:
        # Show reference info if available
        ref_file = f5_config.get('ref_audio') or f5_config.get('ref_file')
        ref_text = f5_config.get('ref_text', "")
        
        # Create base parameters
        base_params = {
            'gen_text': text
        }
        
        # Add reference audio/text from config
        if ref_file:
            base_params['ref_file'] = ref_file  # F5-TTS uses 'ref_file' parameter name
            base_params['ref_text'] = ref_text if ref_text else ""  # Empty string for auto-transcribe
        
        # Use dynamic parameter creation - filters for F5TTS.infer automatically
        generation_params = create_generation_params(
            base_params, 
            f5_config, 
            filter_function=f5tts.infer,
            verbose=f5_config.get('verbose', False)
        )
        
        # Generate audio with all valid parameters
        result = f5tts.infer(**generation_params)
        
        # Handle different return formats from F5-TTS
        if isinstance(result, (tuple, list)) and len(result) >= 2:
            audio, sr = result[0], result[1]
        else:
            audio, sr = result, f5_config.get('sample_rate', 24000)
        
        return audio, sr
        
    except Exception as e:
        print(f"ERROR: F5 generation failed: {e}")
        if f5_config.get('debug_output', False):
            import traceback
            traceback.print_exc(file=sys.stderr)
        return None, None

def save_f5_audio_with_config(audio_data, sample_rate, output_path, f5_config):
    """Save F5 audio with configuration-driven processing"""
    if audio_data is None:
        return False
    
    try:
        # Convert to tensor if needed
        if isinstance(audio_data, torch.Tensor):
            waveform = audio_data
        else:
            waveform = torch.tensor(audio_data, dtype=torch.float32)
        
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Apply audio processing based on config
        if f5_config.get('normalize_audio', True):
            max_val = torch.max(torch.abs(waveform))
            if max_val > 0:
                waveform = waveform / max_val * 0.95
        
        # Apply silence removal if configured
        if f5_config.get('remove_silence', False):
            # Simple silence removal (could be enhanced)
            silence_threshold = 0.01
            non_silent = torch.abs(waveform) > silence_threshold
            if torch.any(non_silent):
                start_idx = torch.argmax(non_silent.int())
                end_idx = len(non_silent) - torch.argmax(torch.flip(non_silent, [0]).int()) - 1
                waveform = waveform[:, start_idx:end_idx+1]
        
        # Apply speed adjustment if configured
        speed = f5_config.get('speed', 1.0)
        if speed != 1.0:
            # Simple speed adjustment by resampling
            target_length = int(waveform.shape[1] / speed)
            if target_length > 0:
                waveform = torch.nn.functional.interpolate(
                    waveform.unsqueeze(0), 
                    size=target_length, 
                    mode='linear', 
                    align_corners=False
                ).squeeze(0)
        
        # Use configured sample rate
        final_sample_rate = f5_config.get('sample_rate', sample_rate)
        
        # Save using torchaudio
        torchaudio.save(str(output_path), waveform.cpu(), final_sample_rate)
        
        if f5_config.get('verbose', False):
            duration = waveform.shape[1] / final_sample_rate
            log_status(f"Saved F5 audio: {output_path.name} ({duration:.1f}s at {final_sample_rate}Hz)")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to save F5 audio: {e}")
        return False

def _update_f5_progress_bar(completed: int, total: int, chunk_times: list):
    """Update horizontal progress bar for F5 chunk processing"""
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
    prefix = "    🎭 F5: "
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

def process_f5_chunks_with_retry(f5tts, chunks, output_dir, f5_config):
    """Process F5 chunks with retry logic, configuration, and progress bar"""
    generated_files = []
    
    # Get retry configuration
    retry_attempts = f5_config.get('retry_attempts', 1)
    retry_delay = f5_config.get('retry_delay', 2.0)
    ignore_errors = f5_config.get('ignore_errors', False)
    skip_failed_chunks = f5_config.get('skip_failed_chunks', False)
    
    chunk_times = []  # Track timing for ETA
    total_chunks = len(chunks)
    
    print(f"  📝 Processing {len(chunks)} chunks with F5-TTS...")
    
    # Print initial progress bar
    _update_f5_progress_bar(0, total_chunks, chunk_times)
    
    for i, chunk_text in enumerate(chunks):
        chunk_num = i + 1
        chunk_start_time = time.time()
        output_file = output_dir / f"chunk_{chunk_num:03d}_f5.wav"
        
        # Retry logic
        success = False
        for attempt in range(retry_attempts):
            try:
                # Generate audio with dynamic configuration
                audio_data, sample_rate = generate_f5_audio_dynamic(f5tts, chunk_text, f5_config)
                
                if audio_data is None:
                    if attempt < retry_attempts - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        print(f"\n    ❌ All {retry_attempts} F5 attempts failed for chunk {chunk_num}")
                        if not ignore_errors:
                            break
                        continue
                
                # Save audio with configuration
                if save_f5_audio_with_config(audio_data, sample_rate, output_file, f5_config):
                    generated_files.append(str(output_file))
                    success = True
                    break
                else:
                    if attempt < retry_attempts - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        print(f"\n    ❌ Failed to save F5 chunk {chunk_num} after {retry_attempts} attempts")
                
            except Exception as e:
                if attempt < retry_attempts - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"\n    ❌ Failed to process F5 chunk {chunk_num} after {retry_attempts} attempts: {e}")
                    if f5_config.get('debug_output', False):
                        import traceback
                        traceback.print_exc(file=sys.stderr)
                    break
        
        if not success and not skip_failed_chunks:
            print(f"\n    ❌ Critical F5 failure on chunk {chunk_num}")
            break
        
        # Record completion time and update progress bar
        chunk_duration = time.time() - chunk_start_time
        chunk_times.append(chunk_duration)
        _update_f5_progress_bar(chunk_num, total_chunks, chunk_times)
    
    # Clear progress bar and show completion
    print()  # New line after progress bar
    print("    ✅ All chunks processed")
    
    return generated_files

def process_f5_text_file(text_file: str, output_dir: str, config: Dict[str, Any], paths: Dict[str, Any]) -> List[str]:
    """Main F5 engine processor with new architecture"""
    if not F5_AVAILABLE:
        raise ImportError("F5-TTS not available. Install with: pip install f5-tts")
    
    try:
        # Extract ALL F5 config parameters dynamically
        f5_config = extract_engine_config(config, 'f5', verbose=True)
        
        # Validate required parameters (all the important F5 params from your config)
        required_params = ['model_type', 'model_name', 'chunk_max_chars', 'speed', 'sample_rate', 
                          'cross_fade_duration', 'sway_sampling_coef', 'cfg_strength', 'nfe_step', 
                          'seed', 'remove_silence', 'ref_text']
        missing_params = validate_required_params(f5_config, required_params, 'f5')
        if missing_params:
            print(f"ERROR: Missing required F5 configuration: {', '.join(missing_params)}")
            return []
        
        log_status(f"Starting F5-TTS processing")
        log_status(f"Model: {f5_config['model_name']}")
        log_status(f"Speed: {f5_config['speed']}x")
        
        # Show important F5 parameters
        important_params = ['cfg_strength', 'nfe_step', 'sway_sampling_coef', 'seed']
        param_values = {k: f5_config[k] for k in important_params if k in f5_config}
        if param_values:
            log_status(f"F5 parameters: {param_values}")
        
        # Auto-detect reference audio and companion text if not explicitly set
        ref_audio = f5_config.get('ref_audio')
        ref_text = f5_config.get('ref_text')
        
        if not ref_audio:
            # Auto-detect from samples directory
            detected_audio, detected_text = auto_detect_reference_audio_and_text(paths)
            if detected_audio:
                f5_config['ref_audio'] = detected_audio
                f5_config['ref_text'] = detected_text if detected_text else ""
                log_status(f"Auto-detected reference audio for F5")
            else:
                log_status(f"No reference audio found, using default F5 voice")
        else:
            # If ref_audio is explicitly set, check for companion text
            if not ref_text:  # Only auto-detect if ref_text is not explicitly set
                detected_text = find_companion_text_file(ref_audio)
                f5_config['ref_text'] = detected_text if detected_text else ""
        
        # Show final voice configuration
        if f5_config.get('ref_audio'):
            log_status(f"Using voice cloning with {Path(f5_config['ref_audio']).name}")
            if f5_config.get('ref_text'):
                log_status(f"Reference text: {f5_config['ref_text'][:50]}{'...' if len(f5_config['ref_text']) > 50 else ''}")
            else:
                log_status(f"Will auto-transcribe reference audio")
        else:
            print("STATUS: No reference audio, using default F5-TTS voice")
        
        # Read clean text (already processed in preprocessing)
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        if not text:
            print(f"ERROR: No text content to process")
            return []
        
        # Load F5 model with configuration
        f5tts = load_f5_model(f5_config)
        if not f5tts:
            return []
        
        # Determine processing strategy based on text length and config
        use_chunking = len(text) > f5_config.get('chunk_max_chars', 350)
        
        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if use_chunking:
            # Process in chunks
            log_status(f"Text length ({len(text)} chars) exceeds chunk limit, using chunked processing")
            chunks = chunk_text_for_f5(text, f5_config)
            log_status(f"Created {len(chunks)} chunks for F5-TTS")
            
            generated_files = process_f5_chunks_with_retry(f5tts, chunks, output_dir, f5_config)
        else:
            # Process entire text as single chunk
            log_status(f"Processing entire text as single F5-TTS generation")
            output_file = output_dir / "complete_f5.wav"
            
            try:
                start_time = time.time()
                
                # Generate audio with dynamic configuration
                audio_data, sample_rate = generate_f5_audio_dynamic(f5tts, text, f5_config)
                
                generation_time = time.time() - start_time
                
                if audio_data is not None:
                    # Save audio using configuration
                    if save_f5_audio_with_config(audio_data, sample_rate, output_file, f5_config):
                        log_status(f"Complete F5 audio generated in {generation_time:.1f}s")
                        generated_files = [str(output_file)]
                    else:
                        print(f"ERROR: Failed to save complete F5 audio")
                        generated_files = []
                else:
                    print(f"ERROR: Failed to generate F5 audio")
                    generated_files = []
                    
            except Exception as e:
                print(f"ERROR: Failed to process F5 text: {e}")
                if f5_config.get('debug_output', False):
                    import traceback
                    traceback.print_exc(file=sys.stderr)
                generated_files = []
        
        # Final statistics
        if use_chunking:
            success_rate = len(generated_files) / len(chunks) * 100 if chunks else 0
            log_status(f"F5-TTS processing completed: {len(generated_files)}/{len(chunks)} files generated ({success_rate:.1f}% success)")
        else:
            success = len(generated_files) > 0
            log_status(f"F5-TTS processing completed: {'Success' if success else 'Failed'}")
        
        if len(generated_files) == 0:
            print(f"ERROR: No F5 audio files were generated successfully")
        
        return generated_files
        
    except Exception as e:
        print(f"ERROR: F5-TTS processing failed: {e}")
        return []

def register_f5_engine():
    """Register F5 engine with the registry"""
    from engines.base_engine import register_engine
    register_engine('f5', process_f5_text_file)
