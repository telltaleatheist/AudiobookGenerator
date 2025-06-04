#!/usr/bin/env python3
"""
XTTS Engine - Coqui XTTS processor with multilingual support
UPDATED: Now uses low-level model interface for full prosody control
Bypasses high-level API limitations to access all generation parameters
"""

import sys
import re
import time
import torch # type: ignore
from pathlib import Path

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

def chunk_text_for_xtts(text, max_chars=400):
    """Split text into chunks optimized for XTTS"""
    # XTTS can handle longer chunks than most other engines
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

def load_xtts_model(xtts_config):
    """Load XTTS model with dynamic configuration support"""
    model_name = xtts_config.get('model_name', 'tts_models/multilingual/multi-dataset/xtts_v2')
    gpu_acceleration = xtts_config.get('gpu_acceleration', True)
    
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
    """Generate audio using XTTS low-level model interface with full prosody control"""
    try:
        # Get the underlying model for low-level access
        model = tts.synthesizer.tts_model
        language = xtts_config.get('language', 'en')
        
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
        
        # Get conditioning latents with dynamic parameters
        gpt_cond_len = xtts_config.get('gpt_cond_len', 30)
        gpt_cond_chunk_len = xtts_config.get('gpt_cond_chunk_len', 4)
        max_ref_len = xtts_config.get('max_ref_len', 60)
        sound_norm_refs = xtts_config.get('sound_norm_refs', True)
        
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
        
        # Add prosody parameters if present in config
        prosody_params = {
            'temperature': 0.75,
            'length_penalty': 1.0,
            'repetition_penalty': 10.0,
            'top_k': 50,
            'top_p': 0.85,
            'do_sample': True,
            'num_beams': 1,
            'speed': 1.0,
            'enable_text_splitting': False
        }
        
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
            'language': xtts_config.get('language', 'en')
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
    """Process chunks with retry logic and enhanced error handling"""
    generated_files = []
    retry_attempts = xtts_config.get('retry_attempts', 1)
    retry_delay = xtts_config.get('retry_delay', 2.0)
    
    for i, chunk_text in enumerate(chunks):
        chunk_num = i + 1
        output_file = output_dir / f"chunk_{chunk_num:03d}_xtts.wav"
        
        print(f"STATUS: Processing chunk {chunk_num}/{len(chunks)} ({len(chunk_text)} chars)", file=sys.stderr)
        
        # Retry logic
        success = False
        for attempt in range(retry_attempts):
            try:
                start_time = time.time()
                
                # Generate audio with low-level interface first, fallback to high-level
                audio_data, sample_rate = generate_xtts_audio_lowlevel(tts, chunk_text, xtts_config)
                
                if audio_data is None:
                    if attempt < retry_attempts - 1:
                        print(f"WARNING: Attempt {attempt + 1} failed, retrying in {retry_delay}s", file=sys.stderr)
                        time.sleep(retry_delay)
                        continue
                    else:
                        print(f"ERROR: All {retry_attempts} attempts failed for chunk {chunk_num}", file=sys.stderr)
                        if not xtts_config.get('ignore_errors', False):
                            break
                        continue
                
                generation_time = time.time() - start_time
                
                # Save audio
                if save_xtts_audio(audio_data, sample_rate, output_file):
                    generated_files.append(str(output_file))
                    print(f"STATUS: Chunk {chunk_num} completed in {generation_time:.1f}s", file=sys.stderr)
                    success = True
                    break
                else:
                    if attempt < retry_attempts - 1:
                        print(f"WARNING: Save failed, retrying...", file=sys.stderr)
                        time.sleep(retry_delay)
                        continue
                    else:
                        print(f"ERROR: Failed to save chunk {chunk_num} after {retry_attempts} attempts", file=sys.stderr)
                
            except Exception as e:
                if attempt < retry_attempts - 1:
                    print(f"ERROR: Attempt {attempt + 1} failed: {e}, retrying...", file=sys.stderr)
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"ERROR: Failed to process chunk {chunk_num} after {retry_attempts} attempts: {e}", file=sys.stderr)
                    break
        
        if not success and not xtts_config.get('skip_failed_chunks', False):
            print(f"ERROR: Critical failure on chunk {chunk_num}", file=sys.stderr)
            break
    
    return generated_files

def process_xtts_text_file(text_file, output_dir, config, paths):
    """Main XTTS engine processor with dynamic configuration detection"""
    if not XTTS_AVAILABLE:
        raise ImportError("XTTS not available. Install with: pip install TTS")
    
    # Extract ALL XTTS config parameters dynamically using registry utilities
    xtts_config = extract_engine_config(config, 'xtts', verbose=True)
    
    print(f"STATUS: Starting XTTS processing (low-level mode)", file=sys.stderr)
    print(f"STATUS: Model: {xtts_config.get('model_name', 'tts_models/multilingual/multi-dataset/xtts_v2')}", file=sys.stderr)
    print(f"STATUS: Language: {xtts_config.get('language', 'en')}", file=sys.stderr)
    if 'speed' in xtts_config:
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
    
    # Chunk text for XTTS
    chunk_max_chars = xtts_config.get('chunk_max_chars', 400)
    chunks = chunk_text_for_xtts(text, chunk_max_chars)
    print(f"STATUS: Created {len(chunks)} chunks for XTTS", file=sys.stderr)
    
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