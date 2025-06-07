#!/usr/bin/env python3
"""
Bark Engine - Fine-tuned Bark TTS processor
UPDATED: Now uses dynamic parameter loading from engine registry
Keeps all existing functionality while adding automatic config detection
"""

import sys
import re
import time
import random
import numpy as np # type: ignore
import gc
from pathlib import Path

# Import dynamic utilities from engine registry
from engines.base_engine import (
    extract_engine_config, 
    filter_params_for_function,
    create_generation_params,
    validate_required_params
)

# Audio processing
try:
    from scipy.io.wavfile import write as write_wav # type: ignore
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: scipy not available for audio saving", file=sys.stderr)

# Bark imports
try:
    import torch # type: ignore
    from bark import SAMPLE_RATE, generate_audio, preload_models # type: ignore
    try:
        from bark.generation import set_seed # type: ignore
        BARK_SET_SEED_AVAILABLE = True
    except ImportError:
        BARK_SET_SEED_AVAILABLE = False
    BARK_AVAILABLE = True
except ImportError:
    BARK_AVAILABLE = False
    print("ERROR: Bark not available. Install with: pip install bark", file=sys.stderr)

def apply_bark_text_preprocessing(text, bark_config):
    """Apply Bark-specific text preprocessing based on dynamic config"""
    processed_text = text
    
    # Handle contractions if configured
    if bark_config.get('expand_contractions', True):
        contractions = {
            "don't": "do not", "won't": "will not", "can't": "cannot",
            "shouldn't": "should not", "wouldn't": "would not",
            "couldn't": "could not", "hasn't": "has not",
            "haven't": "have not", "isn't": "is not", "aren't": "are not",
            "wasn't": "was not", "weren't": "were not",
            "I'm": "I am", "you're": "you are", "we're": "we are",
            "they're": "they are", "it's": "it is", "that's": "that is",
            "here's": "here is", "there's": "there is", "where's": "where is",
            "I'll": "I will", "you'll": "you will", "we'll": "we will",
            "they'll": "they will", "I'd": "I would", "you'd": "you would",
            "we'd": "we would", "they'd": "they would",
            "I've": "I have", "you've": "you have", "we've": "we have",
            "they've": "they have"
        }
        
        for contraction, expansion in contractions.items():
            processed_text = re.sub(r'\b' + re.escape(contraction) + r'\b', 
                                  expansion, processed_text, flags=re.IGNORECASE)
    
    # Remove brackets if configured
    if bark_config.get('remove_brackets', False):
        processed_text = re.sub(r'\[.*?\]', '', processed_text)
        processed_text = re.sub(r'\(.*?\)', '', processed_text)
    
    # Normalize punctuation if configured
    if bark_config.get('normalize_punctuation', True):
        # Replace smart quotes with regular quotes using unicode escapes
        processed_text = re.sub(r'[\u201c\u201d\u201f]', '"', processed_text)  # Smart double quotes
        processed_text = re.sub(r'[\u2018\u2019\u201b]', "'", processed_text)  # Smart single quotes
        processed_text = re.sub(r'—|–', ' -- ', processed_text)
        processed_text = re.sub(r'\.{3,}', '...', processed_text)
    
    # Handle special characters if configured
    if bark_config.get('handle_special_chars', True):
        processed_text = processed_text.replace('&', ' and ')
        processed_text = processed_text.replace('@', ' at ')
        processed_text = processed_text.replace('%', ' percent')
        processed_text = processed_text.replace('#', ' number ')
    
    return processed_text.strip()

def chunk_text_for_bark_dynamic(text, bark_config):
    """Advanced text chunking for Bark with multiple strategies from dynamic config"""
    max_chars = bark_config.get('chunk_max_chars', 150)
    strategy = bark_config.get('chunk_strategy', 'sentence')
    overlap = bark_config.get('overlap_chars', 0)
    
    if strategy == 'paragraph':
        return chunk_by_paragraphs(text, max_chars, overlap)
    elif strategy == 'word_count':
        return chunk_by_word_count(text, max_chars, overlap)
    else:  # Default: sentence
        return chunk_by_sentences(text, max_chars, overlap, bark_config)

def chunk_by_sentences(text, max_chars, overlap, bark_config):
    """Split text by sentences with overlap support"""
    punct = bark_config.get('sentence_split_punct', ['.', '!', '?'])
    pattern = r'(?<=[' + ''.join(re.escape(p) for p in punct) + r'])\s+'
    sentences = re.split(pattern, text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Check if adding this sentence would exceed max
        if current_chunk and len(current_chunk) + len(sentence) + 1 > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                # Add overlap if configured
                if overlap > 0 and len(sentence) <= overlap:
                    current_chunk = sentence
                else:
                    current_chunk = ""
            
            # If sentence itself is too long, force split
            if len(sentence) > max_chars:
                words = sentence.split()
                temp_chunk = ""
                for word in words:
                    if temp_chunk and len(temp_chunk) + len(word) + 1 > max_chars:
                        chunks.append(temp_chunk.strip())
                        temp_chunk = word
                    else:
                        temp_chunk += " " + word if temp_chunk else word
                if temp_chunk:
                    current_chunk = temp_chunk
            else:
                current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def chunk_by_paragraphs(text, max_chars, overlap):
    """Split text by paragraphs with fallback to sentences"""
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        
        if len(paragraph) <= max_chars:
            chunks.append(paragraph)
        else:
            # Paragraph too long, split by sentences
            sub_chunks = chunk_by_sentences(paragraph, max_chars, overlap, {})
            chunks.extend(sub_chunks)
    
    return chunks

def chunk_by_word_count(text, max_chars, overlap):
    """Split text by approximate word count"""
    words = text.split()
    target_words = max_chars // 5  # Rough estimate: 5 chars per word
    overlap_words = overlap // 5
    
    chunks = []
    current_words = []
    
    for word in words:
        current_words.append(word)
        current_text = " ".join(current_words)
        
        if len(current_text) >= max_chars or len(current_words) >= target_words:
            chunks.append(current_text)
            # Apply overlap
            if overlap_words > 0:
                current_words = current_words[-overlap_words:]
            else:
                current_words = []
    
    if current_words:
        chunks.append(" ".join(current_words))
    
    return chunks

def set_bark_seed_dynamic(bark_config, chunk_num=None):
    """Set random seed for Bark generation with dynamic config"""
    seed = bark_config.get('seed')
    
    if seed is not None:
        if bark_config.get('randomize_seed_per_chunk', False) and chunk_num:
            # Use base seed + chunk number for variation
            actual_seed = seed + chunk_num
        else:
            actual_seed = seed
        
        # Set all random seeds
        random.seed(actual_seed)
        np.random.seed(actual_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(actual_seed)
            torch.cuda.manual_seed_all(actual_seed)
        
        # Set Bark's internal seed if available
        if BARK_SET_SEED_AVAILABLE:
            try:
                set_seed(actual_seed)
                print(f"STATUS: Set Bark seed to {actual_seed}", file=sys.stderr)
            except:
                print(f"STATUS: Set system seeds to {actual_seed}", file=sys.stderr)
        else:
            print(f"STATUS: Set system seeds to {actual_seed} (Bark set_seed not available)", file=sys.stderr)

def reload_bark_model_dynamic(bark_config):
    """Enhanced Bark model reloading with dynamic configuration support"""
    print("STATUS: Reloading Bark model to clear context...", file=sys.stderr)
    
    # Clear GPU memory if configured
    if bark_config.get('clear_cuda_cache', True) and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Force garbage collection based on config
    gc_mode = bark_config.get('gc_frequency', 'auto')
    if gc_mode in ['auto', 'chunk']:
        gc.collect()
    
    # Brief pause to let memory clear
    time.sleep(0.5)
    
    # Reload models with configuration
    if bark_config.get('use_smaller_models', False):
        print("STATUS: Smaller models requested (feature not yet implemented)", file=sys.stderr)
    
    preload_models()
    
    # Log memory usage if configured
    if bark_config.get('log_memory_usage', False) and torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1024**3
        print(f"STATUS: GPU memory usage after reload: {memory_used:.2f}GB", file=sys.stderr)
    
    print("STATUS: Bark model reloaded", file=sys.stderr)

def detect_audio_artifacts_dynamic(audio, expected_text, bark_config):
    """Enhanced artifact detection with configurable thresholds from dynamic config"""
    if not bark_config.get('detect_artifacts', True):
        return False
    
    # Configurable thresholds from dynamic config
    duration_threshold = bark_config.get('artifact_threshold', 2.5)
    silence_threshold = bark_config.get('silence_threshold', 0.01)
    max_char_duration = bark_config.get('max_duration_per_char', 0.08)
    
    expected_duration = len(expected_text) * max_char_duration
    actual_duration = len(audio) / SAMPLE_RATE
    
    # Flag suspiciously long audio
    if actual_duration > expected_duration * duration_threshold:
        print(f"WARNING: Audio much longer than expected ({actual_duration:.1f}s vs {expected_duration:.1f}s)", file=sys.stderr)
        return True
    
    # Check for silence
    max_amplitude = np.max(np.abs(audio))
    if max_amplitude < silence_threshold:
        print(f"WARNING: Audio appears silent (max amplitude: {max_amplitude:.4f})", file=sys.stderr)
        return True
    
    # Check for repetition if enabled
    if bark_config.get('repetition_detection', True):
        if detect_repetitive_audio(audio):
            print(f"WARNING: Repetitive audio pattern detected", file=sys.stderr)
            return True
    
    return False

def detect_repetitive_audio(audio, window_size=8000):
    """Detect repetitive patterns in audio"""
    if len(audio) < window_size * 3:
        return False
    
    # Compare the last third with the middle third
    third = len(audio) // 3
    middle_segment = audio[third:2*third]
    last_segment = audio[2*third:2*third + len(middle_segment)]
    
    if len(middle_segment) != len(last_segment):
        return False
    
    # Calculate correlation
    correlation = np.corrcoef(middle_segment, last_segment)[0, 1]
    
    # High correlation suggests repetition
    return correlation > 0.8

def trim_audio_artifacts_dynamic(audio_data, bark_config, sample_rate=24000):
    """Enhanced artifact trimming with configuration support from dynamic config"""
    if not bark_config.get('trim_artifacts', True):
        return audio_data
    
    total_duration = len(audio_data) / sample_rate
    
    # Don't trim very short audio
    if total_duration < 3.0:
        return audio_data
    
    # Use configurable analysis window
    analysis_window = 2.0  # Could be made configurable
    safe_zone_samples = int((total_duration - analysis_window) * sample_rate)
    safe_audio = audio_data[:safe_zone_samples]
    danger_zone = audio_data[safe_zone_samples:]
    
    if len(danger_zone) == 0:
        return audio_data
    
    # Enhanced artifact detection in danger zone
    abs_danger = np.abs(danger_zone)
    max_val = np.max(abs_danger)
    
    if max_val == 0:
        return audio_data
    
    # Configurable speech threshold
    speech_threshold = max_val * 0.03
    window_size = int(0.05 * sample_rate)  # 50ms windows
    
    speech_detected = []
    for i in range(0, len(danger_zone) - window_size, window_size // 2):
        window = abs_danger[i:i + window_size]
        avg_energy = np.mean(window)
        is_speech = avg_energy > speech_threshold
        speech_detected.append(is_speech)
    
    if not speech_detected:
        return audio_data
    
    # Find speech segments
    speech_segments = []
    current_segment_start = None
    
    for i, is_speech in enumerate(speech_detected):
        if is_speech and current_segment_start is None:
            current_segment_start = i
        elif not is_speech and current_segment_start is not None:
            speech_segments.append((current_segment_start, i - 1))
            current_segment_start = None
    
    if current_segment_start is not None:
        speech_segments.append((current_segment_start, len(speech_detected) - 1))
    
    if not speech_segments:
        return audio_data
    
    # Enhanced artifact detection criteria
    if len(speech_segments) >= 2:
        last_segment = speech_segments[-1]
        second_last_segment = speech_segments[-2]
        
        last_start_time = last_segment[0] * (window_size // 2) / sample_rate
        last_duration = (last_segment[1] - last_segment[0]) * (window_size // 2) / sample_rate + 0.05
        second_last_end_time = second_last_segment[1] * (window_size // 2) / sample_rate
        gap_before_last = last_start_time - second_last_end_time
        
        # Configurable artifact criteria
        min_gap = 0.3  # Could be made configurable
        max_fragment_duration = 0.8  # Could be made configurable
        
        if gap_before_last > min_gap and last_duration < max_fragment_duration:
            trim_window = last_segment[0]
            trim_time_in_danger = trim_window * (window_size // 2) / sample_rate
            trim_point = safe_zone_samples + int(trim_time_in_danger * sample_rate)
            
            buffer_samples = int(0.05 * sample_rate)
            trim_point = max(safe_zone_samples, trim_point - buffer_samples)
            
            trimmed_audio = audio_data[:trim_point]
            removed_time = (len(audio_data) - len(trimmed_audio)) / sample_rate
            
            print(f"STATUS: Trimmed {removed_time:.2f}s isolated artifact from end", file=sys.stderr)
            return trimmed_audio
    
    return audio_data

def apply_audio_post_processing_dynamic(audio_data, bark_config):
    """Apply post-processing effects based on dynamic configuration"""
    if not bark_config.get('post_process_audio', True):
        return audio_data
    
    processed_audio = audio_data.copy()
    
    # Normalize audio if configured
    if bark_config.get('normalize_audio', True):
        max_val = np.max(np.abs(processed_audio))
        if max_val > 0:
            processed_audio = processed_audio / max_val * 0.95
    
    # Apply fade effects from dynamic config
    fade_in = bark_config.get('fade_in', 0.0)
    fade_out = bark_config.get('fade_out', 0.05)
    
    if fade_in > 0:
        fade_in_samples = int(fade_in * SAMPLE_RATE)
        if len(processed_audio) > fade_in_samples:
            fade_in_curve = np.linspace(0.0, 1.0, fade_in_samples)
            processed_audio[:fade_in_samples] *= fade_in_curve
    
    if fade_out > 0:
        fade_out_samples = int(fade_out * SAMPLE_RATE)
        if len(processed_audio) > fade_out_samples:
            fade_out_curve = np.linspace(1.0, 0.0, fade_out_samples)
            processed_audio[-fade_out_samples:] *= fade_out_curve
    
    return processed_audio

def generate_bark_audio_dynamic(text, bark_config):
    """Generate audio using Bark with comprehensive dynamic configuration"""
    try:
        # Get voice/history prompt
        voice_prompt = bark_config.get('history_prompt') or bark_config.get('voice', 'v2/en_speaker_0')
        
        # Build base generation parameters
        base_params = {
            'text': text,
            'history_prompt': voice_prompt
        }
        
        # Use dynamic parameter creation - filters for generate_audio automatically
        generation_params = create_generation_params(
            base_params, 
            bark_config, 
            filter_function=generate_audio,
            verbose=True
        )
        
        # Generate audio with all valid parameters
        audio = generate_audio(**generation_params)
        
        return audio
        
    except Exception as e:
        print(f"ERROR: Bark generation failed: {e}", file=sys.stderr)
        return None

def save_bark_audio_dynamic(audio_data, output_path, expected_text, bark_config):
    """Enhanced audio saving with comprehensive post-processing from dynamic config"""
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for audio saving. Install with: pip install scipy")
    
    # Apply artifact trimming
    audio_data = trim_audio_artifacts_dynamic(audio_data, bark_config)
    
    # Apply post-processing
    audio_data = apply_audio_post_processing_dynamic(audio_data, bark_config)
    
    # Save debug information if configured
    if bark_config.get('save_model_outputs', False):
        debug_path = output_path.with_suffix('.debug.npy')
        np.save(debug_path, audio_data)
        print(f"STATUS: Saved debug audio data to {debug_path}", file=sys.stderr)
    
    # Convert to target bit depth from dynamic config
    bit_depth = bark_config.get('bit_depth', 16)
    if bit_depth == 16:
        audio_int = (audio_data * 32767).astype(np.int16)
    elif bit_depth == 24:
        audio_int = (audio_data * 8388607).astype(np.int32)
    else:
        audio_int = (audio_data * 32767).astype(np.int16)  # Default to 16-bit
    
    # Save with configured format
    output_format = bark_config.get('output_format', 'wav')
    if output_format == 'wav':
        write_wav(output_path, SAMPLE_RATE, audio_int)
    else:
        # For other formats, would need additional libraries
        write_wav(output_path, SAMPLE_RATE, audio_int)
        print(f"WARNING: Only WAV format supported, saved as WAV", file=sys.stderr)

def process_bark_text_file(text_file, output_dir, config, paths):
    """Main Bark engine processor with dynamic configuration detection"""
    if not BARK_AVAILABLE:
        raise ImportError("Bark not available. Install with: pip install bark")
    
    # Extract ALL Bark config parameters dynamically using registry utilities
    bark_config = extract_engine_config(config, 'bark', verbose=True)
    
    print(f"STATUS: Starting Bark processing (dynamic mode)", file=sys.stderr)
    print(f"STATUS: Voice: {bark_config.get('voice', 'v2/en_speaker_0')}", file=sys.stderr)
    print(f"STATUS: Temps: text={bark_config.get('text_temp', 0.1)}, waveform={bark_config.get('waveform_temp', 0.15)}", file=sys.stderr)
    
    # Display configured advanced features
    advanced_features = []
    if bark_config.get('use_smaller_models'):
        advanced_features.append("smaller models")
    if bark_config.get('mixed_precision'):
        advanced_features.append("mixed precision")
    if bark_config.get('detect_artifacts'):
        advanced_features.append("artifact detection")
    if bark_config.get('seed') is not None:
        advanced_features.append(f"seed={bark_config['seed']}")
    if bark_config.get('consistency_mode'):
        advanced_features.append("consistency mode")
    
    if advanced_features:
        print(f"STATUS: Advanced features: {', '.join(advanced_features)}", file=sys.stderr)
    
    # Read and preprocess text
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    
    # Apply Bark-specific preprocessing from dynamic config
    processed_text = apply_bark_text_preprocessing(text, bark_config)
    
    if bark_config.get('verbose', False):
        print(f"STATUS: Preprocessed {len(text)} -> {len(processed_text)} characters", file=sys.stderr)
    
    # Advanced chunking with dynamic config
    chunks = chunk_text_for_bark_dynamic(processed_text, bark_config)
    chunk_strategy = bark_config.get('chunk_strategy', 'sentence')
    print(f"STATUS: Created {len(chunks)} chunks using {chunk_strategy} strategy", file=sys.stderr)
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Bark models with configuration
    print("STATUS: Loading Bark models...", file=sys.stderr)
    if bark_config.get('preload_all_models', False):
        print("STATUS: Preloading all Bark models", file=sys.stderr)
    
    preload_models()
    
    # Process chunks with enhanced features from dynamic config
    generated_files = []
    total_chars_processed = 0
    failed_chunks = 0
    
    reload_every_chunks = bark_config.get('reload_model_every_chunks', 15)
    reload_every_chars = bark_config.get('reload_model_every_chars', 2000)
    
    for i, chunk_text in enumerate(chunks):
        chunk_num = i + 1
        
        # Dynamic chunk naming from config
        naming_strategy = bark_config.get('chunk_naming', 'sequential')
        if naming_strategy == 'timestamp':
            timestamp = int(time.time() * 1000)
            output_file = output_dir / f"chunk_{timestamp}_bark.wav"
        elif naming_strategy == 'hash':
            import hashlib
            text_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:8]
            output_file = output_dir / f"chunk_{text_hash}_bark.wav"
        else:  # sequential
            output_file = output_dir / f"chunk_{chunk_num:03d}_bark.wav"
        
        print(f"STATUS: Processing chunk {chunk_num}/{len(chunks)} ({len(chunk_text)} chars)", file=sys.stderr)
        
        # Set seed for this chunk using dynamic config
        set_bark_seed_dynamic(bark_config, chunk_num)
        
        # Check if we should reload model using dynamic config
        total_chars_processed += len(chunk_text)
        should_reload = (
            chunk_num % reload_every_chunks == 0 or
            total_chars_processed >= reload_every_chars
        )
        
        if should_reload and chunk_num > 1:
            reload_bark_model_dynamic(bark_config)
            total_chars_processed = len(chunk_text)
        
        # Retry logic for failed chunks from dynamic config
        retry_attempts = bark_config.get('retry_failed_chunks', 1)
        success = False
        
        for attempt in range(retry_attempts):
            try:
                start_time = time.time()
                
                # Generate audio with dynamic parameters
                audio = generate_bark_audio_dynamic(chunk_text, bark_config)
                
                if audio is None:
                    if attempt < retry_attempts - 1:
                        print(f"WARNING: Generation failed, retrying chunk {chunk_num} (attempt {attempt + 2})", file=sys.stderr)
                        continue
                    else:
                        print(f"ERROR: All {retry_attempts} attempts failed for chunk {chunk_num}", file=sys.stderr)
                        failed_chunks += 1
                        break
                
                generation_time = time.time() - start_time
                
                # Enhanced artifact detection using dynamic config
                has_artifacts = detect_audio_artifacts_dynamic(audio, chunk_text, bark_config)
                if has_artifacts and attempt < retry_attempts - 1:
                    print(f"WARNING: Artifacts detected, retrying chunk {chunk_num} (attempt {attempt + 2})", file=sys.stderr)
                    continue
                
                # Save audio with enhanced processing using dynamic config
                save_bark_audio_dynamic(audio, output_file, chunk_text, bark_config)
                generated_files.append(str(output_file))
                
                print(f"STATUS: Chunk {chunk_num} completed in {generation_time:.1f}s", file=sys.stderr)
                success = True
                break
                
            except Exception as e:
                if attempt < retry_attempts - 1:
                    print(f"ERROR: Chunk {chunk_num} attempt {attempt + 1} failed: {e}, retrying...", file=sys.stderr)
                    time.sleep(1)  # Brief pause before retry
                    continue
                else:
                    print(f"ERROR: Chunk {chunk_num} failed after {retry_attempts} attempts: {e}", file=sys.stderr)
                    failed_chunks += 1
                    
                    # Handle failure based on dynamic config
                    error_mode = bark_config.get('error_recovery_mode', 'retry')
                    if error_mode == 'skip':
                        print(f"STATUS: Skipping failed chunk {chunk_num}", file=sys.stderr)
                        break
                    elif error_mode == 'fallback':
                        fallback_voice = bark_config.get('fallback_voice')
                        if fallback_voice:
                            print(f"STATUS: Trying fallback voice: {fallback_voice}", file=sys.stderr)
                            # Would implement fallback logic here
                        break
                    else:
                        break
        
        if not success and not bark_config.get('skip_failed_chunks', False):
            print(f"ERROR: Critical failure on chunk {chunk_num}, stopping", file=sys.stderr)
            break
    
    # Final statistics
    success_rate = (len(chunks) - failed_chunks) / len(chunks) * 100 if chunks else 0
    print(f"STATUS: Bark processing completed: {len(generated_files)}/{len(chunks)} files generated ({success_rate:.1f}% success)", file=sys.stderr)
    
    if failed_chunks > 0:
        print(f"WARNING: {failed_chunks} chunks failed during processing", file=sys.stderr)
    
    return generated_files

def register_bark_engine():
    """Register Bark engine with the registry"""
    from engines.base_engine import register_engine
    
    # NO DEFAULT CONFIG NEEDED - everything comes from JSON file
    register_engine(
        name='bark',
        processor_func=process_bark_text_file
    )