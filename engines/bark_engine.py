#!/usr/bin/env python3
"""
Bark Engine - Simplified Bark TTS processor
UPDATED: Uses new section-based architecture with essential features only
Focuses on: pronunciation fixes, sentence chunking, basic artifact control
"""

import sys
import re
import time
import random
import numpy as np # type: ignore
import gc
from pathlib import Path
from typing import List, Dict, Any

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

def apply_bark_pronunciation_fixes(text, bark_config):
    """Apply essential pronunciation fixes for Bark"""
    if not bark_config.get('pronunciation_fixes', True):
        return text
    
    # Essential pronunciation fixes for problematic words
    pronunciation_fixes = {
        # Religious/philosophical terms that Bark struggles with
        "atheist": "aytheeist",
        "atheists": "aytheeists", 
        "atheism": "aytheeism",
        "Jehovah's": "jehovas",
        
        # Common problem words
        "colonel": "kernel",
        "hierarchy": "hiyerarkey",
        "epitome": "ihpitomee",
        "hyperbole": "hyperbolee",
        "cache": "cash",
        "niche": "neesh",
        "facade": "fasahd",
        "gauge": "gayj",
        "receipt": "reeseet",
        "height": "hite",
        "leisure": "leezhur",
        
        # Geographic/historical terms
        "bourgeois": "boorzhwah",
        "rendezvous": "rondayvoo",
        "regime": "rehzheem",
        "fascism": "fashism",
        "Nazi": "notsee",
        "Nazis": "notsees",
        "Worcester": "wuster",
        "Leicester": "lester",
        "Arkansas": "arkansaw"
    }
    
    processed_text = text
    fixes_applied = 0
    
    for word, replacement in pronunciation_fixes.items():
        pattern = r'\b' + re.escape(word) + r'\b'
        before_count = len(re.findall(pattern, processed_text, flags=re.IGNORECASE))
        if before_count > 0:
            processed_text = re.sub(pattern, replacement, processed_text, flags=re.IGNORECASE)
            fixes_applied += before_count
    
    if fixes_applied > 0 and bark_config.get('verbose', False):
        print(f"STATUS: Applied {fixes_applied} Bark pronunciation fixes", file=sys.stderr)
    
    return processed_text

def apply_bark_text_preprocessing(text, bark_config):
    """Apply Bark-specific text preprocessing based on config"""
    processed_text = text
    
    # Apply pronunciation fixes
    processed_text = apply_bark_pronunciation_fixes(processed_text, bark_config)
    
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
            "I'll": "I will", "you'll": "you will", "we'll": "we will",
            "they'll": "they will", "I'd": "I would", "you'd": "you would"
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
        # Replace smart quotes with regular quotes
        processed_text = re.sub(r'[\u201c\u201d\u201f]', '"', processed_text)
        processed_text = re.sub(r'[\u2018\u2019\u201b]', "'", processed_text)
        processed_text = re.sub(r'—|–', ' -- ', processed_text)
        processed_text = re.sub(r'\.{3,}', '...', processed_text)
    
    # Handle special characters if configured
    if bark_config.get('handle_special_chars', True):
        processed_text = processed_text.replace('&', ' and ')
        processed_text = processed_text.replace('@', ' at ')
        processed_text = processed_text.replace('%', ' percent')
        processed_text = processed_text.replace('#', ' number ')
    
    return processed_text.strip()

def chunk_text_for_bark(text, bark_config):
    """Simple sentence-based chunking for Bark"""
    max_chars = bark_config.get('chunk_max_chars', 150)
    
    # Split by sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Check if adding this sentence would exceed max
        test_chunk = current_chunk + (" " if current_chunk else "") + sentence
        
        if len(test_chunk) <= max_chars:
            current_chunk = test_chunk
        else:
            # Save current chunk and start new one
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # If sentence itself is too long, force split by words
            if len(sentence) > max_chars:
                words = sentence.split()
                temp_chunk = ""
                for word in words:
                    test_word = temp_chunk + (" " if temp_chunk else "") + word
                    if len(test_word) <= max_chars:
                        temp_chunk = test_word
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = word
                current_chunk = temp_chunk
            else:
                current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def set_bark_seed(bark_config, chunk_num=None):
    """Set random seed for Bark generation"""
    seed = bark_config.get('seed')
    
    if seed is not None:
        if bark_config.get('randomize_seed_per_chunk', False) and chunk_num:
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
                if bark_config.get('verbose', False):
                    print(f"STATUS: Set Bark seed to {actual_seed}", file=sys.stderr)
            except:
                if bark_config.get('verbose', False):
                    print(f"STATUS: Set system seeds to {actual_seed}", file=sys.stderr)

def simple_bark_model_reload(bark_config):
    """Simple Bark model reloading based on config"""
    if bark_config.get('verbose', False):
        print("STATUS: Reloading Bark model...", file=sys.stderr)
    
    # Clear GPU memory if configured
    if bark_config.get('clear_cuda_cache', True) and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Force garbage collection
    gc.collect()
    
    # Brief pause
    time.sleep(0.5)
    
    # Reload models
    preload_models()
    
    if bark_config.get('verbose', False):
        print("STATUS: Bark model reloaded", file=sys.stderr)

def simple_artifact_detection(audio, expected_text, bark_config):
    """Simple artifact detection based on config"""
    if not bark_config.get('detect_artifacts', True):
        return False
    
    # Basic checks using config thresholds
    duration_threshold = bark_config.get('artifact_threshold', 2.5)
    silence_threshold = bark_config.get('silence_threshold', 0.01)
    max_char_duration = bark_config.get('max_duration_per_char', 0.08)
    
    expected_duration = len(expected_text) * max_char_duration
    actual_duration = len(audio) / SAMPLE_RATE
    
    # Flag suspiciously long audio
    if actual_duration > expected_duration * duration_threshold:
        if bark_config.get('verbose', False):
            print(f"WARNING: Audio much longer than expected ({actual_duration:.1f}s vs {expected_duration:.1f}s)", file=sys.stderr)
        return True
    
    # Check for silence
    max_amplitude = np.max(np.abs(audio))
    if max_amplitude < silence_threshold:
        if bark_config.get('verbose', False):
            print(f"WARNING: Audio appears silent (max amplitude: {max_amplitude:.4f})", file=sys.stderr)
        return True
    
    return False

def simple_trim_artifacts(audio_data, bark_config):
    """Simple artifact trimming based on config"""
    if not bark_config.get('trim_artifacts', True):
        return audio_data
    
    # Simple approach: trim last 10% if it looks problematic
    total_length = len(audio_data)
    if total_length < SAMPLE_RATE * 2:  # Don't trim very short audio
        return audio_data
    
    # Analyze last 10%
    trim_point = int(total_length * 0.9)
    main_audio = audio_data[:trim_point]
    end_segment = audio_data[trim_point:]
    
    # Check if end segment has much lower energy (likely artifact)
    main_energy = np.mean(np.abs(main_audio[-SAMPLE_RATE:]))  # Last second of main
    end_energy = np.mean(np.abs(end_segment))
    
    if end_energy < main_energy * 0.1:  # End is much quieter
        if bark_config.get('verbose', False):
            print(f"STATUS: Trimmed low-energy end segment", file=sys.stderr)
        return main_audio
    
    return audio_data

def generate_bark_audio_with_config(text, bark_config):
    """Generate audio using Bark with all config parameters"""
    try:
        # Get voice/history prompt
        voice = bark_config.get('voice', 'v2/en_speaker_0')
        history_prompt = bark_config.get('history_prompt', voice)
        
        # Build generation parameters using all available config
        generation_params = {
            'text': text,
            'history_prompt': history_prompt
        }
        
        # Add all Bark generation parameters from config
        bark_params = ['text_temp', 'waveform_temp', 'silent']
        for param in bark_params:
            if param in bark_config and bark_config[param] is not None:
                generation_params[param] = bark_config[param]
        
        # Use dynamic parameter filtering for generate_audio
        final_params = filter_params_for_function(
            generation_params, 
            generate_audio,
            verbose=bark_config.get('debug_output', False)
        )
        
        if bark_config.get('verbose', False):
            param_summary = {k: v for k, v in final_params.items() if k != 'text'}
            print(f"STATUS: Bark generation params: {param_summary}", file=sys.stderr)
        
        # Generate audio
        audio = generate_audio(**final_params)
        
        return audio
        
    except Exception as e:
        print(f"ERROR: Bark generation failed: {e}", file=sys.stderr)
        return None

def save_bark_audio_with_config(audio_data, output_path, expected_text, bark_config):
    """Save Bark audio with config-driven processing"""
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for audio saving. Install with: pip install scipy")
    
    # Apply artifact trimming if configured
    if bark_config.get('trim_artifacts', True):
        audio_data = simple_trim_artifacts(audio_data, bark_config)
    
    # Apply basic normalization if configured
    if bark_config.get('normalize_audio', True):
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val * 0.95
    
    # Apply fade effects if configured
    if bark_config.get('fade_out', 0.05) > 0:
        fade_samples = int(bark_config['fade_out'] * SAMPLE_RATE)
        if len(audio_data) > fade_samples:
            fade_curve = np.linspace(1.0, 0.0, fade_samples)
            audio_data[-fade_samples:] *= fade_curve
    
    # Convert to target bit depth
    bit_depth = bark_config.get('bit_depth', 16)
    if bit_depth == 16:
        audio_int = (audio_data * 32767).astype(np.int16)
    else:
        audio_int = (audio_data * 32767).astype(np.int16)  # Default to 16-bit
    
    # Save audio
    write_wav(output_path, SAMPLE_RATE, audio_int)
    
    if bark_config.get('verbose', False):
        duration = len(audio_data) / SAMPLE_RATE
        print(f"STATUS: Saved Bark audio: {output_path.name} ({duration:.1f}s)", file=sys.stderr)

def process_bark_chunks_with_retry(chunks, output_dir, bark_config):
    """Process Bark chunks with retry logic and all config features"""
    generated_files = []
    
    # Get configuration
    retry_attempts = bark_config.get('retry_failed_chunks', 3)
    reload_every = bark_config.get('reload_model_every_chunks', 15)
    skip_failed = bark_config.get('skip_failed_chunks', False)
    
    for i, chunk_text in enumerate(chunks):
        chunk_num = i + 1
        output_file = output_dir / f"chunk_{chunk_num:03d}_bark.wav"
        
        print(f"STATUS: Processing Bark chunk {chunk_num}/{len(chunks)} ({len(chunk_text)} chars)", file=sys.stderr)
        
        # Model reload check
        if reload_every > 0 and chunk_num > 1 and (chunk_num - 1) % reload_every == 0:
            simple_bark_model_reload(bark_config)
        
        # Set seed for this chunk
        set_bark_seed(bark_config, chunk_num)
        
        # Retry logic
        success = False
        for attempt in range(retry_attempts):
            try:
                start_time = time.time()
                
                # Generate audio
                audio = generate_bark_audio_with_config(chunk_text, bark_config)
                
                if audio is None:
                    if attempt < retry_attempts - 1:
                        print(f"WARNING: Bark attempt {attempt + 1} failed, retrying...", file=sys.stderr)
                        continue
                    else:
                        print(f"ERROR: All {retry_attempts} Bark attempts failed for chunk {chunk_num}", file=sys.stderr)
                        if not skip_failed:
                            break
                        continue
                
                # Check for artifacts
                has_artifacts = simple_artifact_detection(audio, chunk_text, bark_config)
                if has_artifacts and attempt < retry_attempts - 1:
                    print(f"WARNING: Artifacts detected, retrying chunk {chunk_num}", file=sys.stderr)
                    continue
                
                generation_time = time.time() - start_time
                
                # Save audio
                save_bark_audio_with_config(audio, output_file, chunk_text, bark_config)
                generated_files.append(str(output_file))
                
                print(f"STATUS: Bark chunk {chunk_num} completed in {generation_time:.1f}s", file=sys.stderr)
                success = True
                break
                
            except Exception as e:
                if attempt < retry_attempts - 1:
                    print(f"ERROR: Bark attempt {attempt + 1} failed: {e}, retrying...", file=sys.stderr)
                    time.sleep(1)
                    continue
                else:
                    print(f"ERROR: Failed to process Bark chunk {chunk_num} after {retry_attempts} attempts: {e}", file=sys.stderr)
                    break
        
        if not success and not skip_failed:
            print(f"ERROR: Critical Bark failure on chunk {chunk_num}", file=sys.stderr)
            break
    
    return generated_files

def process_bark_text_file(text_file: str, output_dir: str, config: Dict[str, Any], paths: Dict[str, Any]) -> List[str]:
    """Main Bark engine processor with new architecture (simplified)"""
    if not BARK_AVAILABLE:
        raise ImportError("Bark not available. Install with: pip install bark")
    
    try:
        # Extract ALL Bark config parameters dynamically
        bark_config = extract_engine_config(config, 'bark', verbose=True)
        
        # Validate required parameters
        required_params = ['voice', 'text_temp', 'waveform_temp', 'chunk_max_chars', 'reload_model_every_chunks',
                          'normalize_audio', 'trim_artifacts', 'detect_artifacts', 'retry_failed_chunks', 
                          'skip_failed_chunks', 'verbose', 'debug_output']
        missing_params = validate_required_params(bark_config, required_params, 'bark')
        if missing_params:
            print(f"ERROR: Missing required Bark configuration: {', '.join(missing_params)}", file=sys.stderr)
            return []
        
        print(f"STATUS: Starting Bark processing (simplified)", file=sys.stderr)
        print(f"STATUS: Voice: {bark_config['voice']}", file=sys.stderr)
        print(f"STATUS: Temps: text={bark_config['text_temp']}, waveform={bark_config['waveform_temp']}", file=sys.stderr)
        
        # Show important settings
        important_settings = []
        if bark_config.get('detect_artifacts'):
            important_settings.append("artifact detection")
        if bark_config.get('seed') is not None:
            important_settings.append(f"seed={bark_config['seed']}")
        if bark_config.get('pronunciation_fixes', True):
            important_settings.append("pronunciation fixes")
        
        if important_settings:
            print(f"STATUS: Features: {', '.join(important_settings)}", file=sys.stderr)
        
        # Read and preprocess text
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        if not text:
            print(f"ERROR: No text content to process", file=sys.stderr)
            return []
        
        # Apply Bark-specific preprocessing
        processed_text = apply_bark_text_preprocessing(text, bark_config)
        
        if bark_config.get('verbose', False):
            print(f"STATUS: Preprocessed {len(text)} -> {len(processed_text)} characters", file=sys.stderr)
        
        # Chunk text using simple sentence-based approach
        chunks = chunk_text_for_bark(processed_text, bark_config)
        print(f"STATUS: Created {len(chunks)} chunks using sentence strategy", file=sys.stderr)
        
        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Bark models
        print("STATUS: Loading Bark models...", file=sys.stderr)
        preload_models()
        
        # Process chunks
        generated_files = process_bark_chunks_with_retry(chunks, output_dir, bark_config)
        
        # Final statistics
        success_rate = len(generated_files) / len(chunks) * 100 if chunks else 0
        print(f"STATUS: Bark processing completed: {len(generated_files)}/{len(chunks)} files generated ({success_rate:.1f}% success)", file=sys.stderr)
        
        if len(generated_files) == 0:
            print(f"ERROR: No Bark audio files were generated successfully", file=sys.stderr)
        
        return generated_files
        
    except Exception as e:
        print(f"ERROR: Bark processing failed: {e}", file=sys.stderr)
        return []

def register_bark_engine():
    """Register Bark engine with the registry"""
    from engines.base_engine import register_engine
    register_engine('bark', process_bark_text_file)