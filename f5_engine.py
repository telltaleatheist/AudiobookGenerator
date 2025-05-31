#!/usr/bin/env python3
"""
F5 Engine - F5-TTS processor with natural speech synthesis
FIXED: Simplified to match working f5_test.py approach
"""

import sys
import time
import torch
from pathlib import Path

# F5-TTS imports
try:
    from f5_tts.api import F5TTS
    import torchaudio
    F5_AVAILABLE = True
except ImportError:
    F5_AVAILABLE = False
    print("ERROR: F5-TTS not available. Install with: pip install f5-tts", file=sys.stderr)

def get_f5_default_config():
    """Get default F5-TTS configuration"""
    return {
        'f5': {
            'model_type': 'F5-TTS',
            'model_name': 'F5TTS_Base',
            'ref_audio': None,
            'ref_text': None,
            'speed': 1.0,
            'sample_rate': 24000
        }
    }

def load_f5_model():
    """Load F5-TTS model"""
    print(f"STATUS: Loading F5 model (default)", file=sys.stderr)
    
    try:
        # Initialize F5TTS API with no parameters (uses defaults)
        f5tts = F5TTS()
        
        print("STATUS: F5 model loaded successfully", file=sys.stderr)
        return f5tts
        
    except Exception as e:
        print(f"ERROR: Failed to load F5 model: {e}", file=sys.stderr)
        return None

def generate_f5_audio_simple(f5tts, text, ref_audio=None, ref_text=None, speed=1.0):
    """Generate audio using F5-TTS - simplified approach matching f5_test.py"""
    try:
        print(f"STATUS: Processing entire text in single F5-TTS call", file=sys.stderr)
        print(f"STATUS: Text length: {len(text)} characters", file=sys.stderr)
        
        if ref_audio:
            print(f"STATUS: Using reference audio: {Path(ref_audio).name}", file=sys.stderr)
            if ref_text:
                print(f"STATUS: Using provided reference text", file=sys.stderr)
            else:
                print(f"STATUS: Using empty ref_text (auto-transcribe)", file=sys.stderr)
        
        # IMPORTANT: Force empty ref_text to avoid reference bleeding bug
        # F5-TTS has a known issue where ref_text content bleeds into generated audio
        # Using empty string forces auto-transcription which doesn't have this issue
        result = f5tts.infer(
            ref_file=ref_audio,
            ref_text="",  # Always use empty string to avoid bleeding
            gen_text=text,
            speed=speed
        )
        
        # Handle different return formats from F5-TTS (same as f5_test.py)
        if isinstance(result, (tuple, list)) and len(result) >= 2:
            audio, sr = result[0], result[1]
        else:
            audio, sr = result, 24000
        
        return audio, sr
        
    except Exception as e:
        print(f"ERROR: F5 generation failed: {e}", file=sys.stderr)
        return None, None

def save_f5_audio_simple(audio_data, sample_rate, output_path):
    """Save F5 audio - exactly matching f5_test.py logic"""
    if audio_data is None:
        return False
    
    try:
        # Exact same logic as f5_test.py
        if isinstance(audio_data, torch.Tensor):
            waveform = audio_data
        else:
            waveform = torch.tensor(audio_data, dtype=torch.float32)
        
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Normalize to avoid clipping (exact same as f5_test.py)
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val * 0.95
        
        # Save using torchaudio (exact same as f5_test.py)
        torchaudio.save(str(output_path), waveform.cpu(), sample_rate)
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to save F5 audio: {e}", file=sys.stderr)
        return False

def process_f5_text_file(text_file, output_dir, config, paths):
    """Main F5 engine processor - simplified to match f5_test.py"""
    if not F5_AVAILABLE:
        raise ImportError("F5-TTS not available. Install with: pip install f5-tts")
    
    # Get F5 config
    f5_config = config['f5']
    
    print(f"STATUS: Starting F5-TTS processing (simplified mode)", file=sys.stderr)
    print(f"STATUS: Speed: {f5_config['speed']}x", file=sys.stderr)
    
    # Get reference settings
    ref_audio = f5_config.get('ref_audio')
    ref_text = f5_config.get('ref_text', "")  # Default to empty for auto-transcription
    
    if ref_audio:
        print(f"STATUS: Using voice cloning with {Path(ref_audio).name}", file=sys.stderr)
        if ref_text:
            print(f"STATUS: Reference text: {ref_text[:50]}{'...' if len(ref_text) > 50 else ''}", file=sys.stderr)
        else:
            print(f"STATUS: Will auto-transcribe reference audio", file=sys.stderr)
    else:
        print("STATUS: No reference audio, using default F5-TTS voice", file=sys.stderr)
    
    # Read clean text - NO PRONUNCIATION FIXES (like f5_test.py)
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    
    print(f"STATUS: Processing entire text as single batch ({len(text)} chars)", file=sys.stderr)
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load F5 model
    f5tts = load_f5_model()
    if not f5tts:
        return []
    
    # Process entire text in single batch
    output_file = output_dir / "complete_f5.wav"
    
    try:
        start_time = time.time()
        
        # Generate audio for entire text in one call
        audio_data, sample_rate = generate_f5_audio_simple(
            f5tts,
            text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            speed=f5_config['speed']
        )
        
        generation_time = time.time() - start_time
        
        if audio_data is not None:
            # Save audio using simplified method
            if save_f5_audio_simple(audio_data, sample_rate, output_file):
                print(f"STATUS: Complete audio generated in {generation_time:.1f}s", file=sys.stderr)
                print(f"STATUS: F5 processing completed successfully", file=sys.stderr)
                return [str(output_file)]
            else:
                print(f"ERROR: Failed to save complete audio", file=sys.stderr)
        else:
            print(f"ERROR: Failed to generate audio", file=sys.stderr)
            
    except Exception as e:
        print(f"ERROR: Failed to process text: {e}", file=sys.stderr)
    
    print(f"STATUS: F5 processing failed", file=sys.stderr)
    return []

def register_f5_engine():
    """Register F5 engine with the registry"""
    from engine_registry import register_engine
    
    register_engine(
        name='f5',
        processor_func=process_f5_text_file,
        default_config=get_f5_default_config()
    )