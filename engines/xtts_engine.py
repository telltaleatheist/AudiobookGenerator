#!/usr/bin/env python3
"""
XTTS Engine - Updated for new architecture with no defaults
Clean section-based processing with dynamic parameter loading
"""

import re
import sys
import time
import torch # type: ignore
from pathlib import Path
from typing import List, Dict, Any
from managers.config_manager import ConfigManager, ConfigError

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
    """Main XTTS engine processor - Uses proven chunking strategy from original"""
    if not XTTS_AVAILABLE:
        raise ImportError("XTTS not available. Install with: pip install TTS")
    
    try:
        # Extract and validate XTTS config (new architecture)
        xtts_config = extract_engine_config(config, 'xtts', verbose=True)
        
        # Validate required fields (new architecture)
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
        
        print(f"STATUS: XTTS model: {xtts_config['model_name']}")
        print(f"STATUS: Language: {xtts_config['language']}")
        print(f"STATUS: Speed: {xtts_config['speed']}x")
        print(f"STATUS: Temperature: {xtts_config['temperature']}")
        print(f"STATUS: Repetition penalty: {xtts_config['repetition_penalty']}")
        
        if xtts_config['verbose']:
            print(f"STATUS: All XTTS parameters loaded: {len([k for k, v in xtts_config.items() if v is not None])}")
        
    except (ConfigError, KeyError) as e:
        print(f"❌ XTTS Configuration Error: {e}", file=sys.stderr)
        return []
    
    # Read text
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    
    # Check for voice samples (original method)
    speaker_wav = xtts_config.get('speaker_wav')
    if not speaker_wav:
        detected_audio = auto_detect_reference_audio(paths)
        if detected_audio:
            xtts_config['speaker_wav'] = detected_audio
            speaker_wav = detected_audio
    
    if speaker_wav:
        if isinstance(speaker_wav, list):
            print(f"STATUS: Using {len(speaker_wav)} voice samples")
        else:
            print(f"STATUS: Using voice sample: {Path(speaker_wav).name}")
    else:
        print("❌ XTTS requires voice samples in project/samples/ directory")
        return []
    
    # Use original's proven chunking strategy
    try:
        # Use original's chunking logic - respect XTTS limits
        chunk_max_chars = min(xtts_config['chunk_max_chars'], 249)  # XTTS hard limit
        chunks = smart_dialogue_chunking(text, chunk_max_chars)
        
        print(f"STATUS: Created {len(chunks)} chunks with proven algorithm (max: {chunk_max_chars} chars)")
        
        # Load XTTS model (original method)
        tts = load_xtts_model(xtts_config)
        if not tts:
            return []
        
        # Process chunks with original's retry logic
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generated_files = process_xtts_chunks_with_retry(tts, chunks, output_dir, xtts_config)
        
        print(f"STATUS: XTTS generated {len(generated_files)}/{len(chunks)} files")
        return generated_files
        
    except Exception as e:
        print(f"❌ XTTS processing failed: {e}", file=sys.stderr)
        return []
    
def process_xtts_chunks_with_retry(tts, chunks, output_dir, xtts_config):
    """IMPROVED: Process chunks with smarter gap management"""
    import torch # type: ignore

    generated_files = []
    try:
        retry_attempts = xtts_config['retry_attempts']
        retry_delay = xtts_config['retry_delay']
    except KeyError as e:
        raise ConfigError(f"Missing required XTTS retry configuration: {e}")

    full_audio = []
    final_sample_rate = None

    for i, chunk_text in enumerate(chunks):
        chunk_num = i + 1
        print(f"STATUS: Processing chunk {chunk_num}/{len(chunks)} ({len(chunk_text)} chars)", file=sys.stderr)

        # Retry logic
        success = False
        for attempt in range(retry_attempts):
            try:
                start_time = time.time()

                # Generate audio
                audio_data, sample_rate = generate_xtts_audio_lowlevel(tts, chunk_text, xtts_config)

                if audio_data is None:
                    if attempt < retry_attempts - 1:
                        print(f"WARNING: Attempt {attempt + 1} failed, retrying in {retry_delay}s", file=sys.stderr)
                        time.sleep(retry_delay)
                        continue
                    else:
                        print(f"ERROR: All {retry_attempts} attempts failed for chunk {chunk_num}", file=sys.stderr)
                        if not xtts_config['ignore_errors']:
                            return []
                        continue

                generation_time = time.time() - start_time

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

                # IMPROVED: Smarter gap insertion
                if i < len(chunks) - 1:
                    next_chunk = chunks[i + 1] if i + 1 < len(chunks) else None
                    
                    # Determine gap duration based on content and config
                    silence_duration = determine_gap_type(chunk_text, next_chunk, xtts_config)
                    
                    silence_samples = int(silence_duration * sample_rate)
                    silence = torch.zeros((1, silence_samples), dtype=torch.float32)
                    full_audio.append(silence)

                final_sample_rate = sample_rate
                print(f"STATUS: Chunk {chunk_num} completed in {generation_time:.1f}s", file=sys.stderr)
                success = True
                break

            except Exception as e:
                if attempt < retry_attempts - 1:
                    print(f"ERROR: Attempt {attempt + 1} failed: {e}, retrying...", file=sys.stderr)
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"ERROR: Failed to process chunk {chunk_num} after {retry_attempts} attempts: {e}", file=sys.stderr)
                    break

        if not success and not xtts_config['skip_failed_chunks']:
            print(f"ERROR: Critical failure on chunk {chunk_num}", file=sys.stderr)
            return []

    # Concatenate and save full audio
    if full_audio and final_sample_rate:
        output_path = output_dir / "combined_xtts_output.wav"
        final_waveform = torch.cat(full_audio, dim=1)
        torchaudio.save(str(output_path), final_waveform.cpu(), final_sample_rate)
        print(f"STATUS: Full audio saved to {output_path}", file=sys.stderr)
        generated_files.append(str(output_path))

    return generated_files

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

def generate_xtts_audio_lowlevel(tts, text, xtts_config):
    """Generate audio using XTTS low-level model interface - NO DEFAULTS"""
    try:
        # Get the underlying model for low-level access
        model = tts.synthesizer.tts_model
        
        # No defaults - must be in config
        try:
            language = xtts_config['language']
        except KeyError:
            raise ConfigError("Missing required XTTS configuration: language")
        
        print(f"STATUS: Using low-level XTTS model interface", file=sys.stderr)
        print(f"STATUS: Text length: {len(text)} characters", file=sys.stderr)
        
        # Handle speaker configuration
        speaker_wav = xtts_config.get('speaker_wav')
        if not speaker_wav:
            print("ERROR: speaker_wav required for low-level interface", file=sys.stderr)
            return None, None
        
        if isinstance(speaker_wav, list):
            print(f"STATUS: Using {len(speaker_wav)} reference samples for conditioning", file=sys.stderr)
            # Use first sample as primary, others as additional conditioning
            primary_speaker_wav = speaker_wav[0]
            additional_samples = speaker_wav[1:] if len(speaker_wav) > 1 else []
        else:
            print(f"STATUS: Using single reference sample: {Path(speaker_wav).name}", file=sys.stderr)
            primary_speaker_wav = speaker_wav
            additional_samples = []
        
        try:
            gpt_cond_len = xtts_config['gpt_cond_len']
            gpt_cond_chunk_len = xtts_config['gpt_cond_chunk_len']
            max_ref_len = xtts_config['max_ref_len']
            sound_norm_refs = xtts_config['sound_norm_refs']
        except KeyError as e:
            raise ConfigError(f"Missing required XTTS conditioning configuration: {e}")
        
        print(f"STATUS: Computing conditioning latents (gpt_cond_len={gpt_cond_len}, max_ref_len={max_ref_len})", file=sys.stderr)
        
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
        
        # Show active prosody settings
        active_prosody = {k: v for k, v in prosody_params.items() if k in xtts_config}
        if active_prosody:
            print(f"STATUS: Active prosody settings: {active_prosody}", file=sys.stderr)
        else:
            print(f"STATUS: Using default prosody settings", file=sys.stderr)
        
        # Generate audio using low-level model interface
        print(f"STATUS: Generating audio with full prosody control...", file=sys.stderr)
        
        # Use the model's inference method directly
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
                print(f"STATUS: Model returned dict with keys: {list(result.keys())}", file=sys.stderr)
                # Take the first tensor/array value
                for key, value in result.items():
                    if isinstance(value, (torch.Tensor, list)) and hasattr(value, '__len__'):
                        wav = value
                        print(f"STATUS: Using '{key}' as audio data", file=sys.stderr)
                        break
                else:
                    print(f"ERROR: Could not find audio data in model output", file=sys.stderr)
                    return None, None
        else:
            # Model returned raw audio data
            wav = result
        
        # Get sample rate from synthesizer
        sample_rate = tts.synthesizer.output_sample_rate
        
        print(f"STATUS: Audio generated successfully (format: {type(wav)})", file=sys.stderr)
        return wav, sample_rate
        
    except Exception as e:
        print(f"ERROR: Low-level XTTS generation failed: {e}", file=sys.stderr)
        print(f"ERROR: This might be due to version incompatibility. Falling back to high-level API.", file=sys.stderr)
        
        # Fallback to high-level API
        return generate_xtts_audio_highlevel(tts, text, xtts_config)

def generate_xtts_audio_highlevel(tts, text, xtts_config):
    """Fallback: Generate audio using high-level API (limited prosody control)"""
    try:
        print(f"STATUS: Using high-level XTTS API (limited prosody control)", file=sys.stderr)
        
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
                print(f"STATUS: Using {len(speaker_wav)} reference samples", file=sys.stderr)
            else:
                base_params['speaker_wav'] = speaker_wav
                print(f"STATUS: Using reference sample: {Path(speaker_wav).name}", file=sys.stderr)
        elif xtts_config.get('speaker'):
            base_params['speaker'] = xtts_config['speaker']
            print(f"STATUS: Using built-in speaker: {xtts_config['speaker']}", file=sys.stderr)
        else:
            print("STATUS: No speaker specified, using XTTS default", file=sys.stderr)
        
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

def smart_dialogue_chunking(text, max_chars=250):
    """IMPROVED: Smart chunking within XTTS limit with universal phrase preservation"""
    
    # Normalize text first
    text = re.sub(r'\s+', ' ', text.strip())
    
    chunks = []
    current_chunk = ""
    
    # Split into sentences, preserving dialogue structure
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z"])|(?<=\.\")\s+(?=[A-Z])|(?<=\")\s+(?=[A-Z][a-z])'
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
            # NEW: Check if this break point is safe before splitting
            if current_chunk and is_safe_break_point(current_chunk, sentence):
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                # Try to find a better break point or handle unsafe split
                if current_chunk:
                    # If we're close to the limit but breaking would split a phrase,
                    # try backtracking to find a better split point
                    better_split = find_better_split_point(current_chunk, sentence, max_chars)
                    if better_split:
                        chunks.append(better_split['first_part'])
                        current_chunk = better_split['second_part']
                    else:
                        # No better split found, proceed with original logic
                        chunks.append(current_chunk)
                        current_chunk = sentence
                else:
                    current_chunk = sentence
            
            # If sentence is still too long, split more aggressively
            if len(current_chunk) > max_chars:
                parts = re.split(r'[,;]\s+|(?:\s+(?:and|but|or|however|meanwhile|then|while|when|as|because)\s+)', current_chunk)
                
                if len(parts) > 1:
                    temp_chunk = ""
                    for part in parts:
                        part = part.strip()
                        if not part:
                            continue
                        test_part = temp_chunk + (" " if temp_chunk else "") + part
                        if len(test_part) <= max_chars:
                            temp_chunk = test_part
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk)
                            temp_chunk = part
                            
                            if len(part) > max_chars:
                                words = part.split()
                                temp_chunk = ""
                                for word in words:
                                    test_word = temp_chunk + (" " if temp_chunk else "") + word
                                    if len(test_word) <= max_chars:
                                        temp_chunk = test_word
                                    else:
                                        if temp_chunk:
                                            chunks.append(temp_chunk)
                                        temp_chunk = word
                    current_chunk = temp_chunk
                else:
                    # No punctuation to split on, split by words
                    words = current_chunk.split()
                    current_chunk = ""
                    for word in words:
                        test_word = current_chunk + (" " if current_chunk else "") + word
                        if len(test_word) <= max_chars:
                            current_chunk = test_word
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = word
    
    if current_chunk:
        chunks.append(current_chunk)
    
    # Post-process: merge very short chunks with intelligent boundary checking
    final_chunks = []
    i = 0
    while i < len(chunks):
        chunk = chunks[i]
        
        if len(chunk) < 40 and i < len(chunks) - 1:
            next_chunk = chunks[i + 1]
            merged = chunk + " " + next_chunk
            if len(merged) <= max_chars and is_safe_merge(chunk, next_chunk):
                final_chunks.append(merged)
                i += 2
                continue
        
        final_chunks.append(chunk)
        i += 1
    
    # Safety check: ensure no chunk exceeds max_chars
    verified_chunks = []
    for chunk in final_chunks:
        if len(chunk) <= max_chars:
            verified_chunks.append(chunk)
        else:
            words = chunk.split()
            temp_chunk = ""
            for word in words:
                test = temp_chunk + (" " if temp_chunk else "") + word
                if len(test) <= max_chars:
                    temp_chunk = test
                else:
                    if temp_chunk:
                        verified_chunks.append(temp_chunk)
                    temp_chunk = word
            if temp_chunk:
                verified_chunks.append(temp_chunk)
    
    return verified_chunks

def is_safe_merge(chunk1, chunk2):
    """Check if merging two chunks would create good flow - UNIVERSAL"""
    
    # Don't merge if it would create awkward transitions
    if chunk1.strip().endswith('"') and not chunk2.strip().startswith('"'):
        # Dialogue to narration - usually safe to merge
        return True
    
    if not chunk1.strip().endswith('"') and chunk2.strip().startswith('"'):
        # Narration to dialogue - usually safe to merge
        return True
    
    # Generally safe to merge short chunks
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
            print(f"STATUS: Auto-detected reference audio: {audio_files[0].name}")
            return str(audio_files[0])
        else:
            # Multiple files - return list for XTTS
            print(f"STATUS: Auto-detected {len(audio_files)} reference files")
            return [str(f) for f in audio_files]
        
    except Exception as e:
        print(f"WARNING: Could not auto-detect reference audio: {e}", file=sys.stderr)
        return None

def is_safe_break_point(current_chunk: str, next_sentence: str) -> bool:
    """Check if breaking between chunks would split common phrases"""
    if not current_chunk or not next_sentence:
        return True
    
    current_words = current_chunk.strip().split()
    next_words = next_sentence.strip().split()
    
    if not current_words or not next_words:
        return True
    
    last_word = current_words[-1].lower().rstrip('.,!?";')
    first_word = next_words[0].lower().lstrip('"')
    
    # Don't break if current chunk ends with articles, prepositions
    if last_word in ['a', 'an', 'the', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'to']:
        return False
    
    # Don't break if next sentence starts with conjunctions
    if first_word in ['and', 'but', 'or', 'that', 'which', 'who', 'where', 'when', 'while']:
        return False
    
    # Prefer breaking after complete dialogue
    if current_chunk.strip().endswith('"') or current_chunk.strip().endswith('."'):
        return True
    
    return True

def find_better_split_point(current_chunk, next_sentence, max_chars):
    """Try to find a better split point within current chunk - UNIVERSAL"""
    
    words = current_chunk.split()
    if len(words) <= 2:
        return None
    
    # Try splitting at different points within the current chunk
    for i in range(len(words) - 1, 0, -1):
        first_part = ' '.join(words[:i])
        remaining = ' '.join(words[i:])
        
        # Check if this creates a better split
        if is_safe_break_point(first_part, remaining + ' ' + next_sentence):
            return {
                'first_part': first_part,
                'second_part': remaining + ' ' + next_sentence
            }
    
    return None

def load_xtts_model(xtts_config: Dict[str, Any]):
    """Load XTTS model using ALL configuration parameters"""
    try:
        model_name = xtts_config['model_name']
        
        if xtts_config['verbose']:
            print(f"STATUS: Loading XTTS model: {model_name}")
        
        # Build TTS initialization parameters using ALL relevant config
        init_params = {
            'model_name': model_name,
            'progress_bar': not xtts_config.get('verbose', True),  # Hide progress if not verbose
            'gpu': torch.cuda.is_available()
        }
        
        # Add advanced initialization parameters from config
        advanced_params = ['config_path', 'gpt_model_checkpoint', 'vocoder_checkpoint']
        for param in advanced_params:
            if param in xtts_config and xtts_config[param] is not None:
                init_params[param] = xtts_config[param]
                if xtts_config['verbose']:
                    print(f"STATUS: Using {param}: {xtts_config[param]}")
        
        tts = TTS(**init_params)
        
        if xtts_config['verbose']:
            print("STATUS: XTTS model loaded successfully")
            print(f"STATUS: Sample rate: {xtts_config['sample_rate']}")
            print(f"STATUS: Audio normalization: {xtts_config['normalize_audio']}")
        
        return tts
        
    except Exception as e:
        print(f"❌ Failed to load XTTS model: {e}", file=sys.stderr)
        return None

def process_xtts_chunks_with_reload(tts, chunks: List[str], output_dir: Path, xtts_config: Dict[str, Any]) -> List[str]:
    """Process chunks with model reload logic and ALL config parameters"""
    generated_files = []
    retry_attempts = xtts_config['retry_attempts']
    retry_delay = xtts_config['retry_delay']
    reload_every = xtts_config['reload_model_every_chunks']
    
    full_audio = []
    final_sample_rate = None
    
    # Save intermediate files if configured
    if xtts_config['save_intermediate']:
        intermediate_dir = output_dir / "intermediate"
        intermediate_dir.mkdir(exist_ok=True)
    
    for i, chunk_text in enumerate(chunks):
        chunk_num = i + 1
        
        if xtts_config['verbose']:
            print(f"STATUS: Processing chunk {chunk_num}/{len(chunks)} ({len(chunk_text)} chars)")
        
        # Model reload logic
        if reload_every > 0 and chunk_num > 1 and (chunk_num - 1) % reload_every == 0:
            if xtts_config['verbose']:
                print(f"STATUS: Reloading model after {reload_every} chunks")
            tts = load_xtts_model(xtts_config)
            if not tts:
                print(f"❌ Model reload failed")
                break
        
        # Reset state between chunks if configured
        if xtts_config['reset_state_between_chunks'] and hasattr(tts, 'synthesizer'):
            try:
                if hasattr(tts.synthesizer, 'tts_model'):
                    # Clear any cached states
                    tts.synthesizer.tts_model.inference_cleanup()
            except:
                pass  # Not all models support this
        
        # Retry logic
        success = False
        for attempt in range(retry_attempts):
            try:
                start_time = time.time()
                
                # Generate audio with ALL parameters
                audio_data, sample_rate = generate_xtts_audio_complete(tts, chunk_text, xtts_config)
                
                if audio_data is None:
                    if attempt < retry_attempts - 1:
                        if xtts_config['verbose']:
                            print(f"WARNING: Attempt {attempt + 1} failed, retrying in {retry_delay}s")
                        time.sleep(retry_delay)
                        continue
                    else:
                        print(f"❌ All {retry_attempts} attempts failed for chunk {chunk_num}")
                        if not xtts_config['ignore_errors']:
                            return generated_files
                        continue
                
                generation_time = time.time() - start_time
                
                # Convert to tensor and apply normalization if configured
                if not isinstance(audio_data, torch.Tensor):
                    audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
                else:
                    audio_tensor = audio_data
                
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                
                # Apply normalization if configured
                if xtts_config['normalize_audio']:
                    max_val = torch.max(torch.abs(audio_tensor))
                    if max_val > 0:
                        audio_tensor = audio_tensor / max_val * 0.95
                
                # Save intermediate file if configured
                if xtts_config['save_intermediate']:
                    intermediate_file = intermediate_dir / f"chunk_{chunk_num:03d}.wav"
                    torchaudio.save(str(intermediate_file), audio_tensor.cpu(), sample_rate)
                
                full_audio.append(audio_tensor)
                
                # Add smart silence gap using ALL gap parameters
                if i < len(chunks) - 1:
                    next_chunk = chunks[i + 1] if i + 1 < len(chunks) else None
                    silence_duration = determine_gap_type_complete(chunk_text, next_chunk, xtts_config)
                    
                    silence_samples = int(silence_duration * sample_rate)
                    silence = torch.zeros((1, silence_samples), dtype=torch.float32)
                    full_audio.append(silence)
                
                final_sample_rate = sample_rate
                
                if xtts_config['verbose']:
                    print(f"STATUS: Chunk {chunk_num} completed in {generation_time:.1f}s")
                
                success = True
                break
                
            except Exception as e:
                if attempt < retry_attempts - 1:
                    if xtts_config['verbose']:
                        print(f"ERROR: Attempt {attempt + 1} failed: {e}, retrying...")
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"❌ Failed to process chunk {chunk_num} after {retry_attempts} attempts: {e}")
                    if xtts_config['debug']:
                        import traceback
                        traceback.print_exc()
                    break
        
        if not success and not xtts_config['skip_failed_chunks']:
            print(f"❌ Critical failure on chunk {chunk_num}")
            break
    
    # Concatenate and save full audio using configured sample rate
    if full_audio and final_sample_rate:
        output_path = output_dir / "combined_xtts_output.wav"
        final_waveform = torch.cat(full_audio, dim=1)
        
        # Use configured sample rate
        save_sample_rate = xtts_config.get('sample_rate', final_sample_rate)
        torchaudio.save(str(output_path), final_waveform.cpu(), save_sample_rate)
        
        if xtts_config['verbose']:
            duration = final_waveform.shape[1] / save_sample_rate
            print(f"STATUS: Full audio saved to {output_path} ({duration:.1f}s at {save_sample_rate}Hz)")
        
        generated_files.append(str(output_path))
    
    return generated_files

def generate_xtts_audio_complete(tts, text: str, xtts_config: Dict[str, Any]):
    """Generate audio using XTTS with ALL configuration parameters for entire section"""
    try:
        # Build generation parameters using ALL relevant config
        base_params = {
            'text': text,
            'language': xtts_config['language']
        }
        
        # Add speaker configuration
        speaker_wav = xtts_config.get('speaker_wav')
        if speaker_wav:
            base_params['speaker_wav'] = speaker_wav
        
        # Add ALL XTTS generation parameters from config
        generation_params = [
            'speed', 'temperature', 'length_penalty', 'repetition_penalty',
            'top_k', 'top_p', 'do_sample', 'num_beams', 'enable_text_splitting'
        ]
        
        for param in generation_params:
            if param in xtts_config and xtts_config[param] is not None:
                base_params[param] = xtts_config[param]
        
        # Use dynamic parameter creation with function filtering
        final_params = create_generation_params(
            base_params, 
            xtts_config, 
            filter_function=tts.tts,
            verbose=xtts_config.get('debug', False)
        )
        
        if xtts_config.get('debug', False):
            print(f"DEBUG: XTTS generation params: {final_params}")
        
        # Generate audio for entire section
        wav = tts.tts(**final_params)
        
        # Get sample rate from config or model
        sample_rate = xtts_config.get('sample_rate', 24000)
        if hasattr(tts, 'synthesizer') and hasattr(tts.synthesizer, 'output_sample_rate'):
            sample_rate = tts.synthesizer.output_sample_rate
        
        return wav, sample_rate
        
    except Exception as e:
        if xtts_config.get('debug', False):
            import traceback
            traceback.print_exc()
        print(f"❌ XTTS generation failed: {e}", file=sys.stderr)
        return None, None

def determine_gap_type_complete(current_chunk: str, next_chunk: str, xtts_config: Dict[str, Any]) -> float:
    """Determine appropriate gap using ALL gap configuration parameters"""
    current_text = current_chunk.strip()
    
    # Get ALL gap settings from config
    gap_sentence = xtts_config['silence_gap_sentence']
    gap_dramatic = xtts_config['silence_gap_dramatic'] 
    gap_paragraph = xtts_config['silence_gap_paragraph']
    
    # Very short gaps for dialogue
    if (current_text.endswith('"') or 
        current_text.startswith('"') or
        'said' in current_text.lower()[-20:] or
        'replied' in current_text.lower()[-20:]):
        return 0.3
    
    # Dramatic pauses for emphasis
    if (current_text.endswith('...') or 
        current_text.count('--') > 0 or
        current_text.endswith('!') or
        current_text.endswith('?')):
        return gap_dramatic
    
    # Check for paragraph breaks and scene transitions
    if next_chunk and current_text.endswith('.'):
        next_start = next_chunk.strip()
        
        # Scene transition indicators
        scene_indicators = ['meanwhile', 'later', 'suddenly', 'then', 'however', 
                          'afterwards', 'next', 'finally', 'eventually']
        
        if (next_start.startswith('"') or 
            any(word in next_start.lower()[:50] for word in scene_indicators) or
            (len(next_start) > 0 and next_start[0].isupper())):
            return gap_paragraph
    
    # Normal sentence ending
    if current_text.endswith('.'):
        return gap_sentence
    
    # Default short gap for other punctuation
    return 0.3

def register_xtts_engine():
    """Register XTTS engine"""
    from engines.base_engine import register_engine
    register_engine('xtts', process_xtts_text_file)
