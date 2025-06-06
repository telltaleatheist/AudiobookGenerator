#!/usr/bin/env python3
"""
XTTS Engine - FIXED: Better chunking and prosody preservation
Key fixes:
1. Smarter dialogue-aware chunking
2. Shorter, more consistent silence gaps
3. Better sentence boundary detection
4. Deterministic chunking for consistency
"""

import sys
import re
import time
import torch # type: ignore
from pathlib import Path
from config_manager import ConfigManager, ConfigError

# Import dynamic utilities from engine registry
from engine_registry import (
    extract_engine_config, 
    filter_params_for_function,
    create_generation_params
)

# XTTS imports
try:
    from TTS.api import TTS # type: ignore
    import torchaudio # type: ignore
    XTTS_AVAILABLE = True
except ImportError:
    XTTS_AVAILABLE = False
    print("ERROR: XTTS not available. Install with: pip install TTS", file=sys.stderr)

def auto_detect_reference_audio_and_text(project_paths):
    """Auto-detect reference audio from samples directory (XTTS uses only audio, ignores .txt files)"""
    ref_audio = None
    
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
            print(f"WARNING: Could not locate samples directory", file=sys.stderr)
            return None
    
    if not samples_dir.exists():
        print(f"STATUS: No samples directory found at {samples_dir}", file=sys.stderr)
        return None
    
    # Find all .wav files in samples directory
    wav_files = list(samples_dir.glob("*.wav"))
    if not wav_files:
        print(f"STATUS: No .wav files found in {samples_dir}", file=sys.stderr)
        return None
    
    if len(wav_files) == 1:
        # Single file
        ref_audio = str(wav_files[0])
        print(f"STATUS: Auto-detected reference audio: {wav_files[0].name}", file=sys.stderr)
    else:
        # Multiple files - XTTS supports multiple references
        ref_audio = [str(f) for f in wav_files]
        print(f"STATUS: Auto-detected {len(wav_files)} reference audio files", file=sys.stderr)
        for f in wav_files:
            print(f"  - {f.name}", file=sys.stderr)
    
    return ref_audio

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

def is_safe_break_point(current_chunk, next_sentence):
    """Check if breaking between chunks would split common phrases - UNIVERSAL"""
    
    if not current_chunk or not next_sentence:
        return True
    
    # Get last few words of current chunk and first few words of next
    current_words = current_chunk.strip().split()
    next_words = next_sentence.strip().split()
    
    if not current_words or not next_words:
        return True
    
    last_word = current_words[-1].lower().rstrip('.,!?";')
    first_word = next_words[0].lower().lstrip('"')
    
    # Don't break if current chunk ends with articles
    if last_word in ['a', 'an', 'the']:
        return False
    
    # Don't break if current chunk ends with prepositions
    if last_word in ['of', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'to']:
        return False
    
    # Don't break adjective + noun patterns (like "electromagnetic signature")
    if is_adjective(last_word) and is_noun_like(first_word):
        return False
    
    # Don't break if next sentence starts with conjunctions that continue the thought
    if first_word in ['and', 'but', 'or', 'that', 'which', 'who', 'where', 'when', 'while']:
        return False
    
    # Don't break compound phrases
    if len(current_words) >= 2:
        last_two = ' '.join(current_words[-2:]).lower()
        # Common compound phrase starters
        compound_starters = ['as well', 'such as', 'in order', 'due to', 'rather than', 
                           'more than', 'less than', 'other than', 'as soon']
        for starter in compound_starters:
            if last_two.endswith(starter.split()[-1]) and starter.startswith(last_two.split()[-2]):
                return False
    
    # Prefer breaking after complete dialogue
    if current_chunk.strip().endswith('"') or current_chunk.strip().endswith('."'):
        return True
    
    # Safe to break after sentences that end with finality
    if current_chunk.strip().endswith('.') and not last_word.endswith('.'):
        return True
    
    return True

def is_adjective(word):
    """Simple heuristic to detect adjectives - UNIVERSAL patterns"""
    adjective_endings = [
        'ic', 'al', 'tic', 'ed', 'ing', 'ous', 'ful', 'less', 'ive', 'ible', 'able',
        'ary', 'ory', 'ent', 'ant', 'ish', 'like', 'ly'
    ]
    
    # Remove common punctuation
    clean_word = word.lower().rstrip('.,!?";:')
    
    # Check endings
    for ending in adjective_endings:
        if clean_word.endswith(ending) and len(clean_word) > len(ending) + 2:
            return True
    
    return False

def is_noun_like(word):
    """Simple heuristic to detect nouns - UNIVERSAL patterns"""
    # Remove common punctuation
    clean_word = word.lower().lstrip('"').rstrip('.,!?";:')
    
    # Capitalized words (except sentence starters) are often nouns
    if word[0].isupper() and len(clean_word) > 2:
        return True
    
    # Common noun endings
    noun_endings = ['tion', 'sion', 'ment', 'ness', 'ity', 'ty', 'er', 'or', 'ist', 'ism']
    for ending in noun_endings:
        if clean_word.endswith(ending) and len(clean_word) > len(ending) + 2:
            return True
    
    # Length heuristic - single short words are less likely to be important nouns
    if len(clean_word) >= 4:
        return True
    
    return False

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

def load_xtts_model(xtts_config):
    """Load XTTS model with dynamic configuration support - NO DEFAULTS"""
    
    # No defaults - these must be in config
    try:
        model_name = xtts_config['model_name']
        gpu_acceleration = xtts_config.get('gpu_acceleration', True)  # This one can have a default
    except KeyError as e:
        raise ConfigError(f"Missing required XTTS configuration: {e}")
    
    print(f"STATUS: Loading XTTS model: {model_name}", file=sys.stderr)
    
    try:
        # Build TTS initialization parameters dynamically
        init_params = {
            'model_name': model_name,
            'progress_bar': False,
            'gpu': gpu_acceleration and torch.cuda.is_available()
        }
        
        # Add advanced initialization parameters if present in config
        advanced_init_params = [
            'config_path',
            'gpt_model_checkpoint', 
            'vocoder_checkpoint',
            'use_deepspeed'
        ]
        
        for param in advanced_init_params:
            if param in xtts_config and xtts_config[param] is not None:
                init_params[param] = xtts_config[param]
                print(f"STATUS: Using XTTS init {param}: {xtts_config[param]}", file=sys.stderr)
        
        # Initialize TTS with dynamic parameters
        tts = TTS(**init_params)
        
        print("STATUS: XTTS model loaded successfully", file=sys.stderr)
        print("STATUS: Using low-level model interface for full prosody control", file=sys.stderr)
        return tts
        
    except Exception as e:
        print(f"ERROR: Failed to load XTTS model: {e}", file=sys.stderr)
        return None

def list_xtts_speakers(tts):
    """List available XTTS speakers"""
    try:
        if hasattr(tts, 'speakers') and tts.speakers:
            print("Available XTTS speakers:", file=sys.stderr)
            for i, speaker in enumerate(tts.speakers):
                print(f"  {i}: {speaker}", file=sys.stderr)
            return tts.speakers
        else:
            print("No built-in speakers available for this model", file=sys.stderr)
            return []
    except Exception as e:
        print(f"Warning: Could not list speakers: {e}", file=sys.stderr)
        return []

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

def save_xtts_audio(audio_data, sample_rate, output_path):
    """Save XTTS audio with normalization - matches F5 approach"""
    if audio_data is None:
        return False
    
    try:
        # Convert to tensor if needed (exact same logic as F5)
        if isinstance(audio_data, torch.Tensor):
            waveform = audio_data
        else:
            waveform = torch.tensor(audio_data, dtype=torch.float32)
        
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 2 and waveform.shape[0] > 1:
            # Convert stereo to mono if needed
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Normalize to avoid clipping (exact same as F5)
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val * 0.95
        
        # Save using torchaudio (exact same as F5)
        torchaudio.save(str(output_path), waveform.cpu(), sample_rate)
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to save XTTS audio: {e}", file=sys.stderr)
        return False

def process_xtts_chunks_with_retry(tts, chunks, output_dir, xtts_config):
    """IMPROVED: Process chunks with smarter gap management"""
    import torch

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

def process_xtts_text_file(text_file, output_dir, config, paths):
    """Main XTTS engine processor - NO DEFAULTS VERSION"""
    if not XTTS_AVAILABLE:
        raise ImportError("XTTS not available. Install with: pip install TTS")
    
    # Initialize config manager
    config_manager = ConfigManager()
    
    try:
        # Extract XTTS config and validate required fields
        xtts_config = extract_engine_config(config, 'xtts', verbose=True)
        
        # Validate critical required fields
        required_fields = ['model_name', 'language', 'chunk_max_chars', 'temperature', 
                        'repetition_penalty', 'top_k', 'top_p', 'gpt_cond_len',
                        'retry_attempts', 'retry_delay', 'ignore_errors', 'skip_failed_chunks',
                        'silence_gap_sentence', 'silence_gap_dramatic', 'silence_gap_paragraph']
        
        for field in required_fields:
            if field not in xtts_config:
                raise ConfigError(f"Missing required XTTS configuration: {field}")
        
    except (ConfigError, KeyError) as e:
        print(f"❌ XTTS Configuration Error: {e}", file=sys.stderr)
        print(f"💡 Check your config.json file and ensure all XTTS settings are present", file=sys.stderr)
        print(f"💡 Run: python config_manager.py --create-default", file=sys.stderr)
        return []
    
    # All values below are guaranteed to exist
    print(f"STATUS: Starting XTTS processing (no-defaults mode)", file=sys.stderr)
    print(f"STATUS: Model: {xtts_config['model_name']}", file=sys.stderr)
    print(f"STATUS: Language: {xtts_config['language']}", file=sys.stderr)
    print(f"STATUS: Speed: {xtts_config['speed']}x", file=sys.stderr)
    
    # Display prosody parameters
    prosody_params = ['temperature', 'length_penalty', 'repetition_penalty', 'top_k', 'top_p', 'do_sample']
    active_prosody = {k: xtts_config[k] for k in prosody_params if k in xtts_config}
    if active_prosody:
        print(f"STATUS: Prosody settings: {active_prosody}", file=sys.stderr)
    
    # Display conditioning parameters
    conditioning_params = ['gpt_cond_len', 'gpt_cond_chunk_len', 'max_ref_len', 'sound_norm_refs']
    active_conditioning = {k: xtts_config[k] for k in conditioning_params if k in xtts_config}
    if active_conditioning:
        print(f"STATUS: Conditioning settings: {active_conditioning}", file=sys.stderr)
    
    # Check for voice samples
    speaker_wav = xtts_config.get('speaker_wav')
    if not speaker_wav:
        # Auto-detect from samples directory
        detected_audio = auto_detect_reference_audio_and_text(paths)
        if detected_audio:
            xtts_config['speaker_wav'] = detected_audio  # Update config with detected audio
            speaker_wav = detected_audio
    
    # Display voice cloning info
    if speaker_wav:
        if isinstance(speaker_wav, list):
            print(f"STATUS: Using voice cloning with {len(speaker_wav)} samples", file=sys.stderr)
            for i, sample in enumerate(speaker_wav, 1):
                print(f"  {i}. {Path(sample).name}", file=sys.stderr)
        else:
            print(f"STATUS: Using voice cloning with {Path(speaker_wav).name}", file=sys.stderr)
    elif xtts_config.get('speaker'):
        print(f"STATUS: Using built-in speaker: {xtts_config['speaker']}", file=sys.stderr)
    else:
        print("ERROR: XTTS requires either voice samples or built-in speaker", file=sys.stderr)
        print("💡 Add .wav files to project/samples/ directory", file=sys.stderr)
        return []
    
    # Read clean text (already processed in preprocessing)
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    
    # Use improved chunking
    chunk_max_chars = min(xtts_config['chunk_max_chars'], 249)  # Enforce XTTS limit
    chunks = smart_dialogue_chunking(text, chunk_max_chars)
    
    print(f"STATUS: Created {len(chunks)} chunks with improved algorithm", file=sys.stderr)
    
    # Show chunk boundaries for debugging
    if xtts_config.get('debug', False):
        for i, chunk in enumerate(chunks, 1):
            preview = chunk[:60] + "..." if len(chunk) > 60 else chunk
            print(f"  Chunk {i}: {preview}", file=sys.stderr)
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load XTTS model with dynamic configuration
    tts = load_xtts_model(xtts_config)
    if not tts:
        return []
    
    # List available speakers if no speaker is configured and no samples
    if not speaker_wav and not xtts_config.get('speaker'):
        speakers = list_xtts_speakers(tts)
        if speakers:
            print(f"INFO: Available built-in speakers: {', '.join(speakers[:5])}", file=sys.stderr)
    
    # Process chunks with enhanced retry logic
    generated_files = process_xtts_chunks_with_retry(tts, chunks, output_dir, xtts_config)
    
    # Final statistics
    success_rate = len(generated_files) / len(chunks) * 100 if chunks else 0
    print(f"STATUS: XTTS processing completed: {len(generated_files)}/{len(chunks)} files generated ({success_rate:.1f}% success)", file=sys.stderr)
    
    if len(generated_files) == 0:
        print(f"ERROR: No audio files were generated successfully", file=sys.stderr)
    
    return generated_files

def register_xtts_engine():
    """Register XTTS engine with the registry"""
    from engine_registry import register_engine
    
    # NO DEFAULT CONFIG NEEDED - everything comes from JSON file
    register_engine(
        name='xtts',
        processor_func=process_xtts_text_file
    )