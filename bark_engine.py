#!/usr/bin/env python3
"""
Bark Engine - Fine-tuned Bark TTS processor
Handles pronunciation fixes, model reloading, and artifact detection
"""

import sys
import re
import time
import random
import numpy as np
import gc
from pathlib import Path

# Audio processing
try:
    from scipy.io.wavfile import write as write_wav
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: scipy not available for audio saving", file=sys.stderr)

# Bark imports
try:
    import torch
    from bark import SAMPLE_RATE, generate_audio, preload_models
    BARK_AVAILABLE = True
except ImportError:
    BARK_AVAILABLE = False
    print("ERROR: Bark not available. Install with: pip install bark", file=sys.stderr)

def get_bark_default_config():
    """Get default Bark configuration"""
    return {
        'bark': {
            'voice': 'v2/en_speaker_0',
            'text_temp': 0.1,
            'waveform_temp': 0.15,
            'chunk_max_chars': 200,
            'target_chars': 150,
            'reload_model_every_chunks': 20,
            'reload_model_every_chars': 3000,
            'add_padding': True,
            'padding_format': '. {text}.'
        }
    }

def apply_bark_pronunciation_fixes(text):
    """Apply Bark-specific pronunciation fixes using phonetic spelling"""
    pronunciation_fixes = {
        # Religious/philosophical terms - NO HYPHENS to avoid pauses in Bark
        "atheist": "aytheeist",
        "atheists": "aytheeists", 
        "atheism": "aytheeism",
        "Jehovah's": "jehovas",
        
        # Common problem words - no hyphens/dashes
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
        
        # Religious/historical terms - no hyphens
        "bourgeois": "boorzhwah",
        "rendezvous": "rondayvoo",
        "regime": "rehzheem",
        "fascism": "fashism",
        "Nazi": "notsee",
        "Nazis": "notsees",
        "Aryan": "airy an",
        "pundits": "pundits",
        "ambiguous": "ambigyoous",
        "Christianity": "christianity",
        "religious": "rihliljus",
        
        # Geographic/names - no hyphens
        "Worcester": "wuster",
        "Leicester": "lester",
        "Arkansas": "arkansaw"
    }
    
    fixes_applied = 0
    for word, replacement in pronunciation_fixes.items():
        pattern = r'\b' + re.escape(word) + r'\b'
        before_count = len(re.findall(pattern, text, flags=re.IGNORECASE))
        if before_count > 0:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            fixes_applied += before_count
    
    # Remove problematic names that cause repetition
    text = re.sub(r'\s+Owen Morgan\.?\s*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+[A-Z][a-z]+\s+[A-Z][a-z]+\.?\s*$', '', text)
    
    if fixes_applied > 0:
        print(f"STATUS: Applied {fixes_applied} Bark pronunciation fixes", file=sys.stderr)
    
    return text

def chunk_text_for_bark(text, max_chars=200):
    """Split text into chunks that Bark can handle without truncation"""
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # If adding this sentence would exceed max, start new chunk
        if current_chunk and len(current_chunk) + len(sentence) + 1 > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        
        # If we're at a good size and at sentence boundary, create chunk
        if len(current_chunk) >= max_chars * 0.75:  # 75% of max
            chunks.append(current_chunk.strip())
            current_chunk = ""
    
    # Add remaining text
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Add padding for Bark (helps with quality)
    padded_chunks = []
    for chunk in chunks:
        if chunk:
            padded_chunks.append(f". {chunk}.")
    
    return padded_chunks

def reload_bark_model():
    """Reload Bark model to clear context"""
    print("STATUS: Reloading Bark model to clear context...", file=sys.stderr)
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Force garbage collection
    gc.collect()
    
    # Reset random seeds for deterministic behavior
    seed = random.randint(1, 10000)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Brief pause to let memory clear
    time.sleep(0.5)
    
    # Reload models
    preload_models()
    
    print("STATUS: Bark model reloaded", file=sys.stderr)

def detect_audio_artifacts(audio, expected_text):
    """Detect common Bark audio artifacts"""
    expected_duration = len(expected_text) * 0.08  # ~0.08 seconds per character
    actual_duration = len(audio) / SAMPLE_RATE
    
    # Flag suspiciously long audio (might indicate repetition)
    if actual_duration > expected_duration * 2.5:
        print(f"WARNING: Audio much longer than expected ({actual_duration:.1f}s vs {expected_duration:.1f}s)", file=sys.stderr)
        return True
    
    # Check for silence (generation failure)
    max_amplitude = np.max(np.abs(audio))
    if max_amplitude < 0.01:
        print(f"WARNING: Audio appears silent", file=sys.stderr)
        return True
    
    return False

def trim_audio_artifacts(audio_data, sample_rate=24000):
    """Analyze and trim audio artifacts from the end (preserves your fine-tuning)"""
    total_duration = len(audio_data) / sample_rate
    
    # If audio is less than 3 seconds, don't touch it
    if total_duration < 3.0:
        return audio_data
    
    # Only analyze the last 2 seconds for artifacts
    safe_zone_samples = int((total_duration - 2.0) * sample_rate)
    safe_audio = audio_data[:safe_zone_samples]
    danger_zone = audio_data[safe_zone_samples:]
    
    if len(danger_zone) == 0:
        return audio_data
    
    # Analyze danger zone for artifacts
    abs_danger = np.abs(danger_zone)
    max_val = np.max(abs_danger)
    
    if max_val == 0:
        return audio_data
    
    # Look for the pattern: pause followed by speech fragment
    speech_threshold = max_val * 0.03  # 3% of max in danger zone
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
    
    # Check for isolated artifacts at the end
    if len(speech_segments) >= 2:
        last_segment = speech_segments[-1]
        second_last_segment = speech_segments[-2]
        
        last_start_time = last_segment[0] * (window_size // 2) / sample_rate
        last_duration = (last_segment[1] - last_segment[0]) * (window_size // 2) / sample_rate + 0.05
        second_last_end_time = second_last_segment[1] * (window_size // 2) / sample_rate
        gap_before_last = last_start_time - second_last_end_time
        
        # ARTIFACT CRITERIA: Last segment is isolated and brief
        if gap_before_last > 0.3 and last_duration < 0.8:  # 300ms+ gap, <800ms fragment
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

def save_bark_audio(audio_data, output_path, expected_text=None):
    """Save audio with normalization and artifact trimming"""
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for audio saving. Install with: pip install scipy")
    
    # Trim artifacts (preserves your fine-tuning)
    audio_data = trim_audio_artifacts(audio_data)
    
    # Normalize audio
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val * 0.95
    
    # Light fade-out
    fade_samples = int(0.05 * SAMPLE_RATE)
    if len(audio_data) > fade_samples:
        fade_curve = np.linspace(1.0, 0.0, fade_samples)
        audio_data[-fade_samples:] *= fade_curve
    
    # Convert to 16-bit and save
    audio_int16 = (audio_data * 32767).astype(np.int16)
    write_wav(output_path, SAMPLE_RATE, audio_int16)

def process_bark_text_file(text_file, output_dir, config, paths):
    """Main Bark engine processor"""
    if not BARK_AVAILABLE:
        raise ImportError("Bark not available. Install with: pip install bark")
    
    # Get Bark config
    bark_config = config['bark']
    
    print(f"STATUS: Starting Bark processing", file=sys.stderr)
    print(f"STATUS: Voice: {bark_config['voice']}")
    print(f"STATUS: Temps: text={bark_config['text_temp']}, waveform={bark_config['waveform_temp']}", file=sys.stderr)
    
    # Read clean text
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    
    # Apply Bark-specific processing
    text = apply_bark_pronunciation_fixes(text)
    
    # Chunk text for Bark
    chunks = chunk_text_for_bark(text, bark_config['chunk_max_chars'])
    print(f"STATUS: Created {len(chunks)} chunks for Bark", file=sys.stderr)
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Bark models
    print("STATUS: Loading Bark models...", file=sys.stderr)
    preload_models()
    
    # Process chunks
    generated_files = []
    total_chars_processed = 0
    
    for i, chunk_text in enumerate(chunks):
        chunk_num = i + 1
        output_file = output_dir / f"chunk_{chunk_num:03d}_bark.wav"
        
        print(f"STATUS: Processing chunk {chunk_num}/{len(chunks)} ({len(chunk_text)} chars)", file=sys.stderr)
        
        # Check if we should reload model
        total_chars_processed += len(chunk_text)
        should_reload = (
            chunk_num % bark_config['reload_model_every_chunks'] == 0 or
            total_chars_processed >= bark_config['reload_model_every_chars']
        )
        
        if should_reload and chunk_num > 1:
            reload_bark_model()
            total_chars_processed = len(chunk_text)  # Reset counter
        
        try:
            # Generate audio
            start_time = time.time()
            
            audio = generate_audio(
                chunk_text,
                history_prompt=bark_config['voice'],
                text_temp=bark_config['text_temp'],
                waveform_temp=bark_config['waveform_temp']
            )
            
            generation_time = time.time() - start_time
            
            # Check for artifacts
            if detect_audio_artifacts(audio, chunk_text):
                print(f"WARNING: Possible artifacts detected in chunk {chunk_num}", file=sys.stderr)
            
            # Save audio with artifact trimming
            save_bark_audio(audio, output_file, chunk_text)
            generated_files.append(str(output_file))
            
            print(f"STATUS: Chunk {chunk_num} completed in {generation_time:.1f}s", file=sys.stderr)
            
        except Exception as e:
            print(f"ERROR: Failed to process chunk {chunk_num}: {e}", file=sys.stderr)
            continue
    
    print(f"STATUS: Bark processing completed: {len(generated_files)}/{len(chunks)} files generated", file=sys.stderr)
    return generated_files

def register_bark_engine():
    """Register Bark engine with the registry"""
    from engine_registry import register_engine
    
    register_engine(
        name='bark',
        processor_func=process_bark_text_file,
        default_config=get_bark_default_config()
    )