#!/usr/bin/env python3
"""
F5 Engine - F5-TTS processor with natural speech synthesis
FIXED: Auto-detect companion text files and pass all config settings
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
            # NOTE: ref_text is NOT included in defaults - it's auto-detected from companion files
            'speed': 1.0,
            'sample_rate': 24000,
            # Additional F5-TTS parameters that may be in config
            'cross_fade_duration': 0.15,
            'sway_sampling_coef': -1.0,
            'cfg_strength': 2.0,
            'nfe_step': 32,
            'seed': -1,
            'fix_duration': None,
            'remove_silence': False
        }
    }

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
                print(f"STATUS: Found companion text file: {companion_text_path.name}", file=sys.stderr)
                print(f"STATUS: Companion text ({len(companion_text)} chars): {companion_text[:100]}{'...' if len(companion_text) > 100 else ''}", file=sys.stderr)
                return companion_text
            else:
                print(f"WARNING: Companion text file is empty: {companion_text_path.name}", file=sys.stderr)
                return ""
        except Exception as e:
            print(f"WARNING: Could not read companion text file {companion_text_path.name}: {e}", file=sys.stderr)
            return ""
    
    print(f"STATUS: No companion text file found for {audio_path.name}", file=sys.stderr)
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
            print(f"WARNING: Could not locate samples directory", file=sys.stderr)
            return None, ""
    
    if not samples_dir.exists():
        print(f"STATUS: No samples directory found at {samples_dir}", file=sys.stderr)
        return None, ""
    
    # Find first .wav file in samples directory
    wav_files = list(samples_dir.glob("*.wav"))
    if not wav_files:
        print(f"STATUS: No .wav files found in {samples_dir}", file=sys.stderr)
        return None, ""
    
    # Use first wav file found
    ref_audio = str(wav_files[0])
    print(f"STATUS: Auto-detected reference audio: {wav_files[0].name}", file=sys.stderr)
    
    # Look for companion text file
    ref_text = find_companion_text_file(ref_audio)
    
    return ref_audio, ref_text

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

def generate_f5_audio_simple(f5tts, text, ref_audio=None, ref_text=None, f5_config=None):
    """Generate audio using F5-TTS with all available config parameters"""
    try:
        print(f"STATUS: Processing entire text in single F5-TTS call", file=sys.stderr)
        print(f"STATUS: Text length: {len(text)} characters", file=sys.stderr)
        
        if ref_audio:
            print(f"STATUS: Using reference audio: {Path(ref_audio).name}", file=sys.stderr)
            if ref_text:
                print(f"STATUS: Using companion reference text ({len(ref_text)} chars)", file=sys.stderr)
            else:
                print(f"STATUS: Using empty ref_text (auto-transcribe)", file=sys.stderr)
        
        # Build inference parameters
        infer_params = {
            'ref_file': ref_audio,
            'ref_text': ref_text,  # Use actual ref_text (empty string for auto-transcribe)
            'gen_text': text,
            'speed': f5_config.get('speed', 1.0)
        }
        
        # Add advanced F5-TTS parameters if present in config
        if f5_config:
            advanced_params = [
                'cross_fade_duration', 'sway_sampling_coef', 'cfg_strength', 
                'nfe_step', 'seed', 'fix_duration', 'remove_silence'
            ]
            
            for param in advanced_params:
                if param in f5_config and f5_config[param] is not None:
                    infer_params[param] = f5_config[param]
                    print(f"STATUS: Using F5 {param}: {f5_config[param]}", file=sys.stderr)
        
        # Generate audio
        result = f5tts.infer(**infer_params)
        
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
    """Main F5 engine processor with auto-detection of companion text files"""
    if not F5_AVAILABLE:
        raise ImportError("F5-TTS not available. Install with: pip install f5-tts")
    
    # Get F5 config
    f5_config = config['f5']
    
    print(f"STATUS: Starting F5-TTS processing (simplified mode)", file=sys.stderr)
    print(f"STATUS: Speed: {f5_config['speed']}x", file=sys.stderr)
    
    # Auto-detect reference audio and companion text if not explicitly set
    ref_audio = f5_config.get('ref_audio')
    ref_text = f5_config.get('ref_text')
    
    if not ref_audio:
        # Auto-detect from samples directory
        detected_audio, detected_text = auto_detect_reference_audio_and_text(paths)
        ref_audio = detected_audio
        ref_text = detected_text
    else:
        # If ref_audio is explicitly set, check for companion text
        if not ref_text:  # Only auto-detect if ref_text is not explicitly set
            ref_text = find_companion_text_file(ref_audio)
    
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
            ref_text=ref_text,  # Pass the detected or provided ref_text
            f5_config=f5_config  # Pass full config for advanced parameters
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