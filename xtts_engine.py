#!/usr/bin/env python3
"""
XTTS Engine - Coqui XTTS processor with multilingual support
Handles voice cloning and synthesis with built-in voices
Pronunciation fixes now handled in preprocessing
"""

import sys
import re
import time
import torch # type: ignore
from pathlib import Path

# XTTS imports
try:
    from TTS.api import TTS # type: ignore
    import torchaudio # type: ignore
    XTTS_AVAILABLE = True
except ImportError:
    XTTS_AVAILABLE = False
    print("ERROR: XTTS not available. Install with: pip install TTS", file=sys.stderr)

def get_xtts_default_config():
    """Get default XTTS configuration"""
    return {
        'xtts': {
            'model_name': 'tts_models/multilingual/multi-dataset/xtts_v2',
            'language': 'en',
            'speaker': None,  # For built-in speakers
            'speaker_wav': None,  # Path to speaker reference audio
            'chunk_max_chars': 400,  # XTTS can handle longer chunks
            'target_chars': 300,
            'speed': 1.0,
            'temperature': 0.75,
            'length_penalty': 1.0,
            'repetition_penalty': 5.0,
            'top_k': 50,
            'top_p': 0.85,
            'use_deepspeed': False,
            'gpu_acceleration': True
        }
    }

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

def load_xtts_model(model_name, gpu_acceleration=True):
    """Load XTTS model"""
    print(f"STATUS: Loading XTTS model: {model_name}", file=sys.stderr)
    
    try:
        # Initialize TTS with XTTS model
        tts = TTS(
            model_name=model_name,
            progress_bar=False,
            gpu=gpu_acceleration and torch.cuda.is_available()
        )
        
        print("STATUS: XTTS model loaded successfully", file=sys.stderr)
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

def generate_xtts_audio(tts, text, xtts_config):
    """Generate audio using XTTS"""
    try:
        generation_kwargs = {
            'text': text,
            'language': xtts_config['language'],
            'speed': xtts_config['speed']
        }
        
        # Add speaker configuration
        speaker_wav = xtts_config.get('speaker_wav')
        if speaker_wav:
            # Handle both single file and list of files
            if isinstance(speaker_wav, list):
                # Multiple speaker reference files
                generation_kwargs['speaker_wav'] = speaker_wav
                print(f"STATUS: Using {len(speaker_wav)} reference samples", file=sys.stderr)
            else:
                # Single speaker reference file
                generation_kwargs['speaker_wav'] = speaker_wav
                print(f"STATUS: Using reference sample: {Path(speaker_wav).name}", file=sys.stderr)
        elif xtts_config.get('speaker'):
            # Built-in speaker mode
            generation_kwargs['speaker'] = xtts_config['speaker']
            print(f"STATUS: Using built-in speaker: {xtts_config['speaker']}", file=sys.stderr)
        else:
            print("STATUS: No speaker specified, using XTTS default", file=sys.stderr)
        
        # Add advanced parameters if available
        for param in ['temperature', 'length_penalty', 'repetition_penalty', 'top_k', 'top_p']:
            if param in xtts_config:
                generation_kwargs[param] = xtts_config[param]
        
        # Generate audio
        wav = tts.tts(**generation_kwargs)
        
        return wav, tts.synthesizer.output_sample_rate
        
    except Exception as e:
        print(f"ERROR: XTTS generation failed: {e}", file=sys.stderr)
        if not xtts_config.get('speaker_wav') and not xtts_config.get('speaker'):
            print("💡 Add voice samples to project/samples/ for voice cloning, or use --xtts-speaker for built-in voices", file=sys.stderr)
        return None, None

def save_xtts_audio(audio_data, sample_rate, output_path):
    """Save XTTS audio with normalization"""
    if audio_data is None:
        return False
    
    try:
        # Convert to tensor if needed
        if not isinstance(audio_data, torch.Tensor):
            audio_data = torch.tensor(audio_data, dtype=torch.float32)
        
        # Ensure correct shape
        if audio_data.dim() == 1:
            audio_data = audio_data.unsqueeze(0)
        elif audio_data.dim() == 2 and audio_data.shape[0] > 1:
            # Convert stereo to mono if needed
            audio_data = torch.mean(audio_data, dim=0, keepdim=True)
        
        # Normalize audio
        max_val = torch.max(torch.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val * 0.95
        
        # Save using torchaudio
        torchaudio.save(str(output_path), audio_data, sample_rate)
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to save XTTS audio: {e}", file=sys.stderr)
        return False

def process_xtts_text_file(text_file, output_dir, config, paths):
    """Main XTTS engine processor with automatic sample detection"""
    if not XTTS_AVAILABLE:
        raise ImportError("XTTS not available. Install with: pip install TTS")
    
    # Get XTTS config
    xtts_config = config['xtts']
    
    print(f"STATUS: Starting XTTS processing", file=sys.stderr)
    print(f"STATUS: Model: {xtts_config['model_name']}")
    print(f"STATUS: Language: {xtts_config['language']}", file=sys.stderr)
    print(f"STATUS: Speed: {xtts_config['speed']}x", file=sys.stderr)
    
    # Check for voice samples
    speaker_wav = xtts_config.get('speaker_wav')
    if speaker_wav:
        if isinstance(speaker_wav, list):
            print(f"STATUS: Using {len(speaker_wav)} voice samples for cloning", file=sys.stderr)
        else:
            print(f"STATUS: Using voice sample: {Path(speaker_wav).name}", file=sys.stderr)
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
    chunks = chunk_text_for_xtts(text, xtts_config['chunk_max_chars'])
    print(f"STATUS: Created {len(chunks)} chunks for XTTS", file=sys.stderr)
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load XTTS model
    tts = load_xtts_model(xtts_config['model_name'], xtts_config['gpu_acceleration'])
    if not tts:
        return []
    
    # List available speakers if no speaker is configured and no samples
    if not speaker_wav and not xtts_config.get('speaker'):
        speakers = list_xtts_speakers(tts)
        if speakers:
            print(f"INFO: Available built-in speakers: {', '.join(speakers[:5])}", file=sys.stderr)
    
    # Process chunks
    generated_files = []
    
    for i, chunk_text in enumerate(chunks):
        chunk_num = i + 1
        output_file = output_dir / f"chunk_{chunk_num:03d}_xtts.wav"
        
        print(f"STATUS: Processing chunk {chunk_num}/{len(chunks)} ({len(chunk_text)} chars)", file=sys.stderr)
        
        try:
            start_time = time.time()
            
            # Generate audio
            audio_data, sample_rate = generate_xtts_audio(tts, chunk_text, xtts_config)
            
            generation_time = time.time() - start_time
            
            if audio_data is not None:
                # Save audio
                if save_xtts_audio(audio_data, sample_rate, output_file):
                    generated_files.append(str(output_file))
                    print(f"STATUS: Chunk {chunk_num} completed in {generation_time:.1f}s", file=sys.stderr)
                else:
                    print(f"ERROR: Failed to save chunk {chunk_num}", file=sys.stderr)
            else:
                print(f"ERROR: Failed to generate audio for chunk {chunk_num}", file=sys.stderr)
                
        except Exception as e:
            print(f"ERROR: Failed to process chunk {chunk_num}: {e}", file=sys.stderr)
            continue
    
    print(f"STATUS: XTTS processing completed: {len(generated_files)}/{len(chunks)} files generated", file=sys.stderr)
    return generated_files

def register_xtts_engine():
    """Register XTTS engine with the registry"""
    from engine_registry import register_engine
    
    register_engine(
        name='xtts',
        processor_func=process_xtts_text_file,
        default_config=get_xtts_default_config()
    )