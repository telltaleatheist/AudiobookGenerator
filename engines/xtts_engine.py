#!/usr/bin/env python3
"""
XTTS Engine - Simplified without progress bars
"""

import re
import sys
import time
from core.progress_display_manager import log_error, log_info, log_status
import torch # type: ignore
import warnings
import os
from pathlib import Path
from typing import List, Dict, Any
from managers.config_manager import ConfigManager, ConfigError

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("TTS").setLevel(logging.ERROR)

# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", message=".*GenerationMixin.*")
warnings.filterwarnings("ignore", message=".*prepare_inputs_for_generation.*")
warnings.filterwarnings("ignore", message=".*PreTrainedModel.*")

# Set environment variables to suppress output
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Import dynamic utilities from base engine
from engines.base_engine import (
    extract_engine_config, 
    filter_params_for_function,
    create_generation_params,
    validate_required_params
)

# XTTS imports
try:
    from TTS.api import TTS # type: ignore
    import torchaudio # type: ignore
    XTTS_AVAILABLE = True
except ImportError:
    XTTS_AVAILABLE = False
    print("ERROR: XTTS not available. Install with: pip install TTS", file=sys.stderr)

def process_xtts_text_file(text_file: str, output_dir: str, config: Dict[str, Any], paths: Dict[str, Any]) -> List[str]:
    """Main XTTS engine processor - simplified output"""
    if not XTTS_AVAILABLE:
        raise ImportError("XTTS not available. Install with: pip install TTS")
    
    try:
        # Extract and validate XTTS config
        xtts_config = extract_engine_config(config, 'xtts', verbose=True)
        
        # Validate required fields
        required_fields = [
            'model_name', 'language', 'chunk_max_chars', 'target_chars', 
            'reload_model_every_chunks', 'speed', 'temperature', 'length_penalty',
            'repetition_penalty', 'top_k', 'top_p', 'do_sample', 'num_beams',
            'enable_text_splitting', 'gpt_cond_len', 'gpt_cond_chunk_len', 
            'max_ref_len', 'sound_norm_refs', 'sample_rate', 'normalize_audio',
            'retry_attempts', 'retry_delay', 'ignore_errors', 'skip_failed_chunks',
            'verbose', 'debug', 'save_intermediate', 'silence_gap_sentence', 
            'silence_gap_dramatic', 'silence_gap_paragraph', 'reset_state_between_chunks'
        ]
        
        missing_fields = validate_required_params(xtts_config, required_fields, 'xtts')
        if missing_fields:
            raise ConfigError(f"Missing required XTTS configuration: {', '.join(missing_fields)}")
        
    except (ConfigError, KeyError) as e:
        log_error(f"XTTS Configuration Error: {e}")
        return []
    
    # Read text
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    
    # Check for voice samples
    speaker_wav = xtts_config.get('speaker_wav')
    if not speaker_wav:
        detected_audio = auto_detect_reference_audio(paths)
        if detected_audio:
            xtts_config['speaker_wav'] = detected_audio
            speaker_wav = detected_audio
    
    if not speaker_wav:
        print("❌ XTTS requires voice samples in project/samples/ directory")
        return []
    
    # Use proven chunking strategy
    try:
        # Respect XTTS limits
        chunk_max_chars = min(xtts_config['chunk_max_chars'], 249)  # XTTS hard limit
        chunks = smart_dialogue_chunking(text, chunk_max_chars)
        
        print(f"  📝 Processing {len(chunks)} chunks...")
        
        # Load XTTS model
        tts = load_xtts_model(xtts_config)
        if not tts:
            return []
        
        # Process chunks with retry logic
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generated_files = process_xtts_chunks_with_retry(tts, chunks, output_dir, xtts_config)
        
        return generated_files
        
    except Exception as e:
        log_error(f"XTTS processing failed: {e}")
        return []

def score_split_point(first_part: str, second_part: str) -> float:
    """Score a potential split point based on linguistic quality"""
    score = 0.0
    
    first_words = first_part.strip().split()
    second_words = second_part.strip().split()
    
    if not first_words or not second_words:
        return -1.0
    
    last_word = first_words[-1].lower().rstrip('.,!?";:')
    first_word = second_words[0].lower().lstrip('"')
    
    # Positive scoring factors
    if first_part.strip().endswith('.'):
        score += 2.0  # Sentence boundary
    
    if first_part.strip().endswith('"'):
        score += 1.5  # Dialogue boundary
    
    if first_part.strip().endswith(('!', '?')):
        score += 1.0  # Strong punctuation
    
    # Negative scoring factors
    if last_word in ['a', 'an', 'the']:
        score -= 3.0  # Articles
    
    if last_word in ['of', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'to']:
        score -= 2.0  # Prepositions
    
    if first_word in ['and', 'but', 'or', 'that', 'which']:
        score -= 1.5  # Conjunctions
    
    # Length factors
    if len(first_part) < 30:
        score -= 1.0  # Too short
    
    return score

def handle_oversized_chunk_enhanced(chunk: str, max_chars: int, existing_chunks: list) -> str:
    """Enhanced handling of chunks that are still too long"""
    
    if len(chunk) <= max_chars:
        return chunk
    
    # Try splitting on various punctuation marks, prioritized by quality
    split_patterns = [
        r'[,;]\s+',  # Commas and semicolons
        r'(?:\s+(?:and|but|or|however|meanwhile|then|while|when|as|because)\s+)',  # Conjunctions
        r'(?:\s+(?:after|before|during|since|until|while)\s+)',  # Time conjunctions
        r'\s+--\s+',  # Em dashes
        r'\s+\(\s*',  # Parentheses
    ]
    
    for pattern in split_patterns:
        parts = re.split(pattern, chunk)
        if len(parts) > 1:
            result_chunks = []
            temp_chunk = ""
            
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                
                test_chunk = temp_chunk + (" " if temp_chunk else "") + part
                if len(test_chunk) <= max_chars:
                    temp_chunk = test_chunk
                else:
                    if temp_chunk:
                        result_chunks.append(temp_chunk)
                    temp_chunk = part
            
            if temp_chunk:
                result_chunks.append(temp_chunk)
            
            # Add all but the last chunk to existing_chunks
            for chunk_part in result_chunks[:-1]:
                existing_chunks.append(chunk_part)
            
            # Return the last chunk for continued processing
            return result_chunks[-1] if result_chunks else ""
    
    # If no good split points found, fall back to word splitting
    words = chunk.split()
    temp_chunk = ""
    for word in words:
        test_word = temp_chunk + (" " if temp_chunk else "") + word
        if len(test_word) <= max_chars:
            temp_chunk = test_word
        else:
            if temp_chunk:
                existing_chunks.append(temp_chunk)
            temp_chunk = word
    
    return temp_chunk

def validate_chunk_quality(chunks: list, max_chars: int) -> list:
    """Final quality validation and cleanup"""
    
    validated_chunks = []
    
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        
        # Ensure chunk length compliance
        if len(chunk) <= max_chars:
            validated_chunks.append(chunk)
        else:
            # Emergency word-based splitting for non-compliant chunks
            words = chunk.split()
            temp_chunk = ""
            for word in words:
                test = temp_chunk + (" " if temp_chunk else "") + word
                if len(test) <= max_chars:
                    temp_chunk = test
                else:
                    if temp_chunk:
                        validated_chunks.append(temp_chunk)
                    temp_chunk = word
            if temp_chunk:
                validated_chunks.append(temp_chunk)
    
    return validated_chunks

def post_process_chunks_enhanced(chunks: list, max_chars: int) -> list:
    """Enhanced post-processing with smarter merging decisions"""
    
    if not chunks:
        return chunks
    
    final_chunks = []
    i = 0
    
    while i < len(chunks):
        chunk = chunks[i]
        
        # Try to merge very short chunks with next chunk
        if len(chunk) < 40 and i < len(chunks) - 1:
            next_chunk = chunks[i + 1]
            merged = chunk + " " + next_chunk
            
            if len(merged) <= max_chars and is_safe_merge(chunk, next_chunk):
                final_chunks.append(merged)
                i += 2
                continue
        
        # Try to merge with previous chunk if current is very short
        elif len(chunk) < 25 and final_chunks:
            prev_chunk = final_chunks[-1]
            merged = prev_chunk + " " + chunk
            
            if len(merged) <= max_chars and is_safe_merge(prev_chunk, chunk):
                final_chunks[-1] = merged
                i += 1
                continue
        
        final_chunks.append(chunk)
        i += 1
    
    return final_chunks

def process_xtts_chunks_with_retry(tts, chunks, output_dir, xtts_config):
    """Process chunks with horizontal progress bar"""
    import torch # type: ignore

    generated_files = []
    try:
        retry_attempts = xtts_config['retry_attempts']
        retry_delay = xtts_config['retry_delay']
    except KeyError as e:
        raise ConfigError(f"Missing required XTTS retry configuration: {e}")

    full_audio = []
    final_sample_rate = None
    
    total_chunks = len(chunks)
    chunk_times = []  # Track timing for ETA
    
    # Print initial progress bar
    _update_chunk_progress_bar(0, total_chunks, chunk_times)

    for i, chunk_text in enumerate(chunks):
        chunk_num = i + 1
        chunk_start_time = time.time()

        # Retry logic
        success = False
        for attempt in range(retry_attempts):
            try:
                # Redirect stdout/stderr to suppress model warnings during generation
                import contextlib
                import io
                
                # Capture all output during audio generation
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    # Generate audio
                    audio_data, sample_rate = generate_xtts_audio_lowlevel(tts, chunk_text, xtts_config)

                if audio_data is None:
                    if attempt < retry_attempts - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        print(f"    ❌ All {retry_attempts} attempts failed for chunk {chunk_num}")
                        if not xtts_config['ignore_errors']:
                            return []
                        continue

                # Normalize and convert to tensor
                if not isinstance(audio_data, torch.Tensor):
                    audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
                else:
                    audio_tensor = audio_data

                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)  # (1, N)

                max_val = torch.max(torch.abs(audio_tensor))
                if max_val > 0:
                    audio_tensor = audio_tensor / max_val * 0.95

                full_audio.append(audio_tensor)

                # Smarter gap insertion
                if i < len(chunks) - 1:
                    next_chunk = chunks[i + 1] if i + 1 < len(chunks) else None
                    
                    # Determine gap duration based on content and config
                    silence_duration = determine_gap_type(chunk_text, next_chunk, xtts_config)
                    
                    silence_samples = int(silence_duration * sample_rate)
                    silence = torch.zeros((1, silence_samples), dtype=torch.float32)
                    full_audio.append(silence)

                final_sample_rate = sample_rate
                success = True
                break

            except Exception as e:
                if attempt < retry_attempts - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"\n    ❌ Failed to process chunk {chunk_num} after {retry_attempts} attempts: {e}")
                    break

        if not success and not xtts_config['skip_failed_chunks']:
            print(f"\n    ❌ Critical failure on chunk {chunk_num}")
            return []

        # Record completion time and update progress bar
        chunk_duration = time.time() - chunk_start_time
        chunk_times.append(chunk_duration)
        _update_chunk_progress_bar(chunk_num, total_chunks, chunk_times)

    # Clear progress bar and show completion
    print()  # New line after progress bar
    print("    ✅ All chunks processed")

    # Concatenate and save full audio
    if full_audio and final_sample_rate:
        output_path = output_dir / "combined_xtts_output.wav"
        final_waveform = torch.cat(full_audio, dim=1)
        torchaudio.save(str(output_path), final_waveform.cpu(), final_sample_rate)
        generated_files.append(str(output_path))

    return generated_files

def _update_chunk_progress_bar(completed: int, total: int, chunk_times: list):
    """Update horizontal progress bar for chunk processing"""
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
    prefix = "    🎤 TTS: "
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

def generate_xtts_audio_lowlevel(tts, text, xtts_config):
    """Generate audio using XTTS low-level model interface"""
    try:
        # Get the underlying model for low-level access
        model = tts.synthesizer.tts_model
        
        # No defaults - must be in config
        try:
            language = xtts_config['language']
        except KeyError:
            raise ConfigError("Missing required XTTS configuration: language")
        
        # Handle speaker configuration
        speaker_wav = xtts_config.get('speaker_wav')
        if not speaker_wav:
            print("ERROR: speaker_wav required for low-level interface", file=sys.stderr)
            return None, None
        
        if isinstance(speaker_wav, list):
            # Use first sample as primary, others as additional conditioning
            primary_speaker_wav = speaker_wav[0]
            additional_samples = speaker_wav[1:] if len(speaker_wav) > 1 else []
        else:
            primary_speaker_wav = speaker_wav
            additional_samples = []
        
        try:
            gpt_cond_len = xtts_config['gpt_cond_len']
            gpt_cond_chunk_len = xtts_config['gpt_cond_chunk_len']
            max_ref_len = xtts_config['max_ref_len']
            sound_norm_refs = xtts_config['sound_norm_refs']
        except KeyError as e:
            raise ConfigError(f"Missing required XTTS conditioning configuration: {e}")
        
        # Build reference audio list
        ref_audio_paths = [primary_speaker_wav] + additional_samples
        
        # Get conditioning latents from the model
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=ref_audio_paths,
            gpt_cond_len=gpt_cond_len,
            gpt_cond_chunk_len=gpt_cond_chunk_len,
            max_ref_length=max_ref_len,
            sound_norm_refs=sound_norm_refs
        )
        
        # Build generation parameters with full prosody control
        generation_params = {
            'text': text,
            'language': language,
            'gpt_cond_latent': gpt_cond_latent,
            'speaker_embedding': speaker_embedding,
        }
        
        try:
            prosody_params = {
                'temperature': xtts_config['temperature'],
                'length_penalty': xtts_config['length_penalty'],
                'repetition_penalty': xtts_config['repetition_penalty'],
                'top_k': xtts_config['top_k'],
                'top_p': xtts_config['top_p'],
                'do_sample': xtts_config['do_sample'],
                'num_beams': xtts_config['num_beams'],
                'speed': xtts_config['speed'],
                'enable_text_splitting': xtts_config['enable_text_splitting']
            }
        except KeyError as e:
            raise ConfigError(f"Missing required XTTS prosody configuration: {e}")
        
        # Override with config values
        for param in prosody_params:
            if param in xtts_config:
                prosody_params[param] = xtts_config[param]
        
        # Add all prosody parameters to generation
        generation_params.update(prosody_params)
        
        # Generate audio using low-level model interface
        result = model.inference(**generation_params)
        
        # Handle different return formats from low-level model
        if isinstance(result, dict):
            # Model returned a dictionary - extract audio data
            if 'wav' in result:
                wav = result['wav']
            elif 'audio' in result:
                wav = result['audio']
            else:
                # Try to find audio data in the dict
                for key, value in result.items():
                    if isinstance(value, (torch.Tensor, list)) and hasattr(value, '__len__'):
                        wav = value
                        break
                else:
                    print("ERROR: Could not find audio data in model output", file=sys.stderr)
                    return None, None
        else:
            # Model returned raw audio data
            wav = result
        
        # Get sample rate from synthesizer
        sample_rate = tts.synthesizer.output_sample_rate
        
        return wav, sample_rate
        
    except Exception as e:
        print(f"ERROR: Low-level XTTS generation failed: {e}", file=sys.stderr)
        
        # Fallback to high-level API
        return generate_xtts_audio_highlevel(tts, text, xtts_config)
            
def generate_xtts_audio_highlevel(tts, text, xtts_config):
    """Fallback: Generate audio using high-level API"""
    try:
        # Build base generation parameters for high-level API
        base_params = {
            'text': text,
            'language': xtts_config['language']
        }

        # Only add speed if it's actually in the config
        if 'speed' in xtts_config:
            base_params['speed'] = xtts_config['speed']

        # Add speaker configuration
        speaker_wav = xtts_config.get('speaker_wav')
        if speaker_wav:
            # Handle both single file and list of files
            if isinstance(speaker_wav, list):
                base_params['speaker_wav'] = speaker_wav
            else:
                base_params['speaker_wav'] = speaker_wav
        elif xtts_config.get('speaker'):
            base_params['speaker'] = xtts_config['speaker']
        
        # Use dynamic parameter creation - filters for TTS.tts automatically
        generation_params = create_generation_params(
            base_params, 
            xtts_config, 
            filter_function=tts.tts,
            verbose=True
        )
        
        # Generate audio with all valid parameters
        wav = tts.tts(**generation_params)
        
        return wav, tts.synthesizer.output_sample_rate
        
    except Exception as e:
        print(f"ERROR: High-level XTTS generation failed: {e}", file=sys.stderr)
        return None, None

def load_xtts_model(xtts_config: Dict[str, Any]):
    """Load XTTS model - suppress all output"""
    try:
        model_name = xtts_config['model_name']
        
        # Build TTS initialization parameters using ALL relevant config
        init_params = {
            'model_name': model_name,
            'progress_bar': False,  # Always hide progress bar
            'gpu': torch.cuda.is_available()
        }
        
        # Add advanced initialization parameters from config
        advanced_params = ['config_path', 'gpt_model_checkpoint', 'vocoder_checkpoint']
        for param in advanced_params:
            if param in xtts_config and xtts_config[param] is not None:
                init_params[param] = xtts_config[param]
        
        # Suppress all output during model loading
        import contextlib
        import io
        
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            tts = TTS(**init_params)
        
        return tts
        
    except Exception as e:
        log_info(f"Failed to load XTTS model: {e}", "error")
        return None

def determine_gap_type(current_chunk, next_chunk=None, xtts_config=None):
    """Determine appropriate gap based on content and config settings - NO DEFAULTS"""
    current_text = current_chunk.strip()
    
    if not xtts_config:
        raise ConfigError("Missing xtts_config for gap determination")
    
    # Get gap settings from config - all are required, no defaults
    try:
        gap_sentence = xtts_config['silence_gap_sentence']
        gap_dramatic = xtts_config['silence_gap_dramatic'] 
        gap_paragraph = xtts_config['silence_gap_paragraph']
    except KeyError as e:
        raise ConfigError(f"Missing XTTS silence gap configuration: {e}")
    
    # Very short gaps for dialogue
    if (current_text.endswith('"') or 
        current_text.startswith('"') or
        'said' in current_text.lower()[-20:] or
        'replied' in current_text.lower()[-20:]):
        return 0.3
    
    # Check for paragraph breaks
    if next_chunk and (current_text.endswith('.') and 
                      (next_chunk.strip().startswith('"') or 
                       len(next_chunk.strip()) > 0 and next_chunk.strip()[0].isupper())):
        if any(word in next_chunk.lower()[:50] for word in ['meanwhile', 'later', 'suddenly', 'then']):
            return gap_paragraph
    
    # Dramatic pauses
    if (current_text.endswith('...') or 
        current_text.count('--') > 0 or
        current_text.endswith('!') or
        current_text.endswith('?')):
        return gap_dramatic
    
    # Normal sentence ending
    if current_text.endswith('.'):
        return gap_sentence
    
    # Default short gap
    return 0.3

def smart_dialogue_chunking(text, max_chars=250):
    """ENHANCED: Advanced chunking with better phrase preservation and dialogue handling"""
    
    # Normalize text first
    text = re.sub(r'\s+', ' ', text.strip())
    
    chunks = []
    current_chunk = ""
    
    # Enhanced sentence splitting that preserves dialogue structure better
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z"])|(?<=\.\")\s+(?=[A-Z])|(?<=\")\s+(?=[A-Z][a-z])|(?<=\!\")\s+|(?<=\?\")\s+'
    sentences = re.split(sentence_pattern, text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Check if adding this sentence would exceed the limit
        test_chunk = current_chunk + (" " if current_chunk else "") + sentence
        
        if len(test_chunk) <= max_chars:
            current_chunk = test_chunk
        else:
            # ENHANCED: Better break point analysis
            if current_chunk and is_safe_break_point(current_chunk, sentence):
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                # Try to find a better break point within current chunk
                if current_chunk:
                    better_split = find_better_split_point(current_chunk, sentence, max_chars)
                    if better_split:
                        chunks.append(better_split['first_part'])
                        current_chunk = better_split['second_part']
                    else:
                        # If no better split found, use original logic but with warnings
                        if len(current_chunk) > max_chars * 0.8:  # Only split if we're near the limit
                            chunks.append(current_chunk)
                            current_chunk = sentence
                        else:
                            # Try to keep them together if we're not too close to limit
                            current_chunk = test_chunk
                else:
                    current_chunk = sentence
            
            # Enhanced aggressive splitting for overly long chunks
            if len(current_chunk) > max_chars:
                current_chunk = handle_oversized_chunk_enhanced(current_chunk, max_chars, chunks)
    
    if current_chunk:
        chunks.append(current_chunk)
    
    # ENHANCED: Post-process with smarter merging and quality validation
    final_chunks = post_process_chunks_enhanced(chunks, max_chars)
    
    # Final quality validation
    validated_chunks = validate_chunk_quality(final_chunks, max_chars)
    
    return validated_chunks

def is_safe_merge(chunk1, chunk2):
    """Check if merging two chunks would create good flow - UNIVERSAL"""
    
    chunk1_stripped = chunk1.strip()
    chunk2_stripped = chunk2.strip()
    
    # Dialogue transition rules
    if chunk1_stripped.endswith('"') and not chunk2_stripped.startswith('"'):
        return True  # Dialogue to narration
    
    if not chunk1_stripped.endswith('"') and chunk2_stripped.startswith('"'):
        return True  # Narration to dialogue
    
    # Don't merge if both are dialogue from potentially different speakers
    if chunk1_stripped.endswith('"') and chunk2_stripped.startswith('"'):
        return False
    
    # Check for topic shifts
    if any(word in chunk2_stripped.lower()[:50] for word in ['meanwhile', 'later', 'suddenly', 'however', 'nevertheless', 'on the other hand']):
        return False  # Likely topic shift
    
    return True

def auto_detect_reference_audio(paths: Dict[str, Any]) -> str:
    """Auto-detect reference audio from samples directory"""
    try:
        if 'project_dir' in paths:
            samples_dir = Path(paths['project_dir']) / 'samples'
        else:
            # Fallback: look for samples relative to batch dir
            batch_dir = Path(paths['batch_dir'])
            samples_dir = batch_dir.parent.parent / 'samples'
        
        if not samples_dir.exists():
            return None
        
        # Find all audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(samples_dir.glob(f"*{ext}"))
        
        if not audio_files:
            return None
        
        if len(audio_files) == 1:
            return str(audio_files[0])
        else:
            # Multiple files - return list for XTTS
            return [str(f) for f in audio_files]
        
    except Exception as e:
        print(f"WARNING: Could not auto-detect reference audio: {e}", file=sys.stderr)
        return None

def is_safe_break_point(current_chunk: str, next_sentence: str) -> bool:
    """Enhanced break point analysis with better linguistic awareness"""
    
    if not current_chunk or not next_sentence:
        return True
    
    current_words = current_chunk.strip().split()
    next_words = next_sentence.strip().split()
    
    if not current_words or not next_words:
        return True
    
    last_word = current_words[-1].lower().rstrip('.,!?";:')
    first_word = next_words[0].lower().lstrip('"')
    
    # Enhanced linguistic rules
    
    # 1. Don't break articles and determiners
    if last_word in ['a', 'an', 'the', 'this', 'that', 'these', 'those', 'some', 'any', 'every', 'each']:
        return False
    
    # 2. Don't break prepositions (expanded list)
    prepositions = ['of', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'to', 'into', 'onto', 
                   'upon', 'under', 'over', 'through', 'between', 'among', 'during', 'before', 
                   'after', 'above', 'below', 'across', 'around', 'behind', 'beside']
    if last_word in prepositions:
        return False
    
    # 3. Don't break conjunctions that continue thoughts
    if first_word in ['and', 'but', 'or', 'that', 'which', 'who', 'where', 'when', 'while', 
                     'although', 'because', 'since', 'unless', 'until', 'whereas']:
        return False
    
    # 4. Enhanced adjective-noun detection
    if is_adjective(last_word) and is_noun_like(first_word):
        return False
    
    # 5. Don't break compound phrases (enhanced detection)
    if len(current_words) >= 2:
        last_two = ' '.join(current_words[-2:]).lower()
        compound_phrases = [
            'as well', 'such as', 'in order', 'due to', 'rather than', 'more than', 
            'less than', 'other than', 'as soon', 'so that', 'even though', 'as if',
            'in case', 'on behalf', 'according to', 'in spite', 'regardless of'
        ]
        for phrase in compound_phrases:
            if last_two.endswith(phrase.split()[-1]) and any(last_two.startswith(phrase.split()[0]) for phrase in compound_phrases):
                return False
    
    # 6. Prefer breaking after complete dialogue
    current_stripped = current_chunk.strip()
    if current_stripped.endswith('"') or current_stripped.endswith('."') or current_stripped.endswith('!"') or current_stripped.endswith('?"'):
        return True
    
    # 7. Prefer breaking after complete sentences
    if current_stripped.endswith('.') and not last_word.endswith('.'):
        return True
    
    # 8. Don't break mid-dialogue
    if '"' in current_chunk and not current_stripped.endswith('"'):
        # We're in the middle of dialogue
        return False
    
    return True

def is_adjective(word: str) -> bool:
    """Simple heuristic to detect adjectives - UNIVERSAL patterns"""
    adjective_endings = [
        'ic', 'al', 'tic', 'ed', 'ing', 'ous', 'ful', 'less', 'ive', 'ible', 'able',
        'ary', 'ory', 'ent', 'ant', 'ish', 'like', 'ly', 'ese', 'ine', 'ile'
    ]
    
    clean_word = word.lower().rstrip('.,!?";:')
    
    if len(clean_word) < 3:
        return False
    
    for ending in adjective_endings:
        if clean_word.endswith(ending) and len(clean_word) > len(ending) + 2:
            return True
    
    # Common adjectives that don't follow patterns
    common_adjectives = [
        'good', 'bad', 'big', 'small', 'old', 'new', 'long', 'short', 'high', 'low',
        'hot', 'cold', 'warm', 'cool', 'fast', 'slow', 'hard', 'soft', 'strong', 'weak'
    ]
    
    return clean_word in common_adjectives

def is_noun_like(word):
    """Simple heuristic to detect nouns - UNIVERSAL patterns"""
    clean_word = word.lower().lstrip('"').rstrip('.,!?";:')
    
    # Capitalized words (except sentence starters) are often nouns
    if word[0].isupper() and len(clean_word) > 2:
        return True
    
    # Common noun endings
    noun_endings = [
        'tion', 'sion', 'ment', 'ness', 'ity', 'ty', 'er', 'or', 'ist', 'ism',
        'ure', 'age', 'ance', 'ence', 'ship', 'hood', 'dom', 'ward'
    ]
    
    for ending in noun_endings:
        if clean_word.endswith(ending) and len(clean_word) > len(ending) + 2:
            return True
    
    # Length and pattern heuristics
    if len(clean_word) >= 4 and not clean_word.endswith(('ly', 'ing', 'ed')):
        return True
    
    return False

def find_better_split_point(current_chunk, next_sentence, max_chars):
    """Try to find a better split point within current chunk - UNIVERSAL"""
    
    words = current_chunk.split()
    if len(words) <= 2:
        return None
    
    # Try splitting at different points, prioritizing good break locations
    best_split = None
    best_score = -1
    
    for i in range(len(words) - 1, max(1, len(words) - 10), -1):  # Don't go too far back
        first_part = ' '.join(words[:i])
        remaining = ' '.join(words[i:])
        
        if len(first_part) < max_chars * 0.6:  # Don't make chunks too small
            continue
        
        # Score this split point
        score = score_split_point(first_part, remaining + ' ' + next_sentence)
        
        if score > best_score:
            best_score = score
            best_split = {
                'first_part': first_part,
                'second_part': remaining + ' ' + next_sentence
            }
    
    return best_split

def register_xtts_engine():
    """Register XTTS engine"""
    from engines.base_engine import register_engine
    register_engine('xtts', process_xtts_text_file)