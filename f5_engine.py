#!/usr/bin/env python3
"""
F5 Engine - FIXED: ONLY use config settings that are explicitly present
No defaults, no fallbacks - config file is the single source of truth
"""

import sys
import time
import re
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
    """REMOVED: No longer provides defaults - config file is single source of truth"""
    return {
        'f5': {
            # NO DEFAULTS - user must specify everything they want in config.json
        }
    }

def auto_detect_reference_audio_and_text(paths):
    """Auto-detect reference audio and text from samples directory"""
    # Get project directory from paths
    if 'project_dir' in paths:
        samples_dir = Path(paths['project_dir']) / 'samples'
    else:
        # Fallback: derive from other paths
        for path_key, path_value in paths.items():
            if isinstance(path_value, (str, Path)):
                potential_project = Path(path_value).parent.parent
                samples_dir = potential_project / 'samples'
                if samples_dir.exists():
                    break
        else:
            print(f"ERROR: Could not locate samples directory", file=sys.stderr)
            return None, None
    
    if not samples_dir.exists():
        print(f"ERROR: Samples directory not found: {samples_dir}", file=sys.stderr)
        print(f"F5-TTS requires reference audio. Please add a .wav file to the samples directory.", file=sys.stderr)
        return None, None
    
    # Find .wav files in samples directory
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(list(samples_dir.glob(f"*{ext}")))
        audio_files.extend(list(samples_dir.glob(f"*{ext.upper()}")))
    
    if not audio_files:
        print(f"ERROR: No audio files found in {samples_dir}", file=sys.stderr)
        print(f"F5-TTS requires reference audio. Please add a .wav/.mp3 file to the samples directory.", file=sys.stderr)
        return None, None
    
    # Use first audio file found (sorted alphabetically)
    audio_files = sorted(set(audio_files), key=lambda p: p.name.lower())
    ref_audio = str(audio_files[0])
    
    print(f"STATUS: Auto-detected reference audio: {audio_files[0].name}", file=sys.stderr)
    
    # Look for companion text file with same stem
    text_file = audio_files[0].with_suffix('.txt')
    if text_file.exists():
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                ref_text = f.read().strip()
            print(f"STATUS: Found companion text file: {text_file.name} ({len(ref_text)} chars)", file=sys.stderr)
            return ref_audio, ref_text
        except Exception as e:
            print(f"WARNING: Could not read companion text file {text_file.name}: {e}", file=sys.stderr)
    
    # No text file - use auto-transcription
    print(f"STATUS: No companion text file found - will auto-transcribe audio", file=sys.stderr)
    return ref_audio, ""

def process_f5_text_file(text_file, output_dir, config, paths):
    """Main F5 engine processor - ONLY use config settings that are present"""
    if not F5_AVAILABLE:
        raise ImportError("F5-TTS not available. Install with: pip install f5-tts")
    
    # Get F5 config from config file
    f5_config = config.get('f5', {})
    
    if not f5_config:
        raise ValueError("No F5 config section found in config file")
    
    print(f"STATUS: F5 using ONLY settings from config file", file=sys.stderr)
    print(f"STATUS: Found {len(f5_config)} F5 parameters in config", file=sys.stderr)
    
    # Log what parameters we found
    if f5_config:
        print(f"STATUS: F5 config parameters: {list(f5_config.keys())}", file=sys.stderr)
    
    # Get reference audio and text - ONLY use config if explicitly set
    ref_audio = f5_config.get('ref_audio')
    ref_text = f5_config.get('ref_text', "")
    
    if ref_audio and Path(ref_audio).exists():
        print(f"STATUS: Using ref_audio from config: {Path(ref_audio).name}", file=sys.stderr)
        
        # FIXED: If ref_text is empty, look for companion .txt file
        if not ref_text.strip():
            print(f"STATUS: ref_text in config is empty, looking for companion .txt file", file=sys.stderr)
            companion_txt = Path(ref_audio).with_suffix('.txt')
            if companion_txt.exists():
                try:
                    with open(companion_txt, 'r', encoding='utf-8') as f:
                        ref_text = f.read().strip()
                    print(f"STATUS: Found companion text file: {companion_txt.name} ({len(ref_text)} chars)", file=sys.stderr)
                except Exception as e:
                    print(f"WARNING: Could not read companion text file: {e}", file=sys.stderr)
                    print(f"STATUS: Will use auto-transcription", file=sys.stderr)
            else:
                print(f"STATUS: No companion .txt file found, will use auto-transcription", file=sys.stderr)
        else:
            print(f"STATUS: Using ref_text from config: '{ref_text[:50]}...' ({len(ref_text)} chars)", file=sys.stderr)
    else:
        if ref_audio:
            print(f"WARNING: Config ref_audio not found: {ref_audio}", file=sys.stderr)
        else:
            print(f"STATUS: No ref_audio in config", file=sys.stderr)
        
        print(f"STATUS: Auto-detecting reference audio from samples/", file=sys.stderr)
        ref_audio, ref_text = auto_detect_reference_audio_and_text(paths)
        
        if not ref_audio:
            raise ValueError("F5-TTS requires reference audio. Set ref_audio in config or add .wav file to samples/")
    
    # Read text to process
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    
    # Determine processing strategy - ONLY from config, NO defaults
    text_length = len(text)
    smart_chunking = f5_config.get('smart_chunking')  # No default
    chunk_max_chars = f5_config.get('chunk_max_chars')  # No default
    
    # Use chunking ONLY if explicitly enabled in config
    use_chunking = bool(smart_chunking) or (chunk_max_chars and text_length > chunk_max_chars)
    
    print(f"STATUS: Text length: {text_length} chars", file=sys.stderr)
    print(f"STATUS: smart_chunking from config: {smart_chunking}", file=sys.stderr)
    print(f"STATUS: chunk_max_chars from config: {chunk_max_chars}", file=sys.stderr)
    print(f"STATUS: Will use chunking: {use_chunking}", file=sys.stderr)
    
    if use_chunking:
        return process_f5_chunked(text, output_dir, f5_config, ref_audio, ref_text)
    else:
        return process_f5_single(text, output_dir, f5_config, ref_audio, ref_text)

def preprocess_text_for_f5(text, f5_config):
    """Preprocess text - ONLY if explicitly enabled in config"""
    
    # Check what preprocessing is enabled in config - NO defaults
    normalize_numbers = f5_config.get('normalize_numbers')
    add_pause_markers = f5_config.get('add_pause_markers')
    smart_chunking = f5_config.get('smart_chunking')
    
    # Only preprocess if EXPLICITLY enabled in config
    if not any([normalize_numbers, add_pause_markers, smart_chunking]):
        print("STATUS: No text preprocessing enabled in config", file=sys.stderr)
        return text
    
    print(f"STATUS: Text preprocessing enabled:", file=sys.stderr)
    print(f"  normalize_numbers: {normalize_numbers}", file=sys.stderr)
    print(f"  add_pause_markers: {add_pause_markers}", file=sys.stderr)
    print(f"  smart_chunking: {smart_chunking}", file=sys.stderr)
    
    processed_text = text
    
    # 1. Normalize numbers if enabled
    if normalize_numbers:
        number_map = {
            '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five',
            '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten',
            '11': 'eleven', '12': 'twelve', '13': 'thirteen', '14': 'fourteen',
            '15': 'fifteen', '16': 'sixteen', '17': 'seventeen', '18': 'eighteen',
            '19': 'nineteen', '20': 'twenty'
        }
        
        for num, word in number_map.items():
            processed_text = re.sub(r'\b' + num + r'\b', word, processed_text)
    
    # 2. Add pause markers if enabled
    if add_pause_markers:
        processed_text = re.sub(r'\s+(and|but|or|so|yet|for|nor)\s+', r', \1 ', processed_text)
        processed_text = re.sub(r'^(However|Therefore|Nevertheless|Furthermore|Moreover|Additionally|Meanwhile|Finally|First|Second|Third|Last)\s+', r'\1, ', processed_text, flags=re.MULTILINE)
        processed_text = re.sub(r'\s*([,.!?;:])\s*', r'\1 ', processed_text)
        processed_text = re.sub(r'\s+', ' ', processed_text)
    
    # 3. Smart sentence breaking if enabled
    if smart_chunking:
        long_sentence_pattern = r'([^.!?]{100,}?)(\s+(?:and|but|or|so|because|since|while|although|however|therefore)\s+)'
        processed_text = re.sub(long_sentence_pattern, r'\1.\2', processed_text, flags=re.IGNORECASE)
        processed_text = re.sub(r'([^.!?]{150,}?)(\s+[A-Z])', r'\1. \2', processed_text)
    
    return processed_text.strip()

def chunk_text_for_f5(text, f5_config):
    """Chunk text using ONLY config file settings - NO defaults"""
    
    # Get chunking parameters from config ONLY - NO defaults
    chunk_max_chars = f5_config.get('chunk_max_chars')
    smart_chunking = f5_config.get('smart_chunking')
    
    if not chunk_max_chars:
        print(f"WARNING: No chunk_max_chars in config, using single chunk", file=sys.stderr)
        return [text]
    
    print(f"STATUS: Chunking with max_chars={chunk_max_chars}, smart={smart_chunking}", file=sys.stderr)
    
    # Preprocess if enabled
    processed_text = preprocess_text_for_f5(text, f5_config)
    
    if not smart_chunking:
        # Simple character-based chunking
        chunks = []
        words = processed_text.split()
        current_chunk = ""
        
        for word in words:
            if len(current_chunk) + len(word) + 1 > chunk_max_chars and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = word
            else:
                current_chunk += " " + word if current_chunk else word
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        print(f"STATUS: Simple chunking created {len(chunks)} chunks", file=sys.stderr)
        return chunks
    
    else:
        # Smart chunking at sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', processed_text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if current_chunk and len(current_chunk) + len(sentence) + 1 > chunk_max_chars:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
            
            if len(current_chunk) >= chunk_max_chars * 0.8:
                chunks.append(current_chunk.strip())
                current_chunk = ""
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        avg_length = sum(len(c) for c in chunks) // len(chunks) if chunks else 0
        print(f"STATUS: Smart chunking created {len(chunks)} chunks (avg: {avg_length} chars)", file=sys.stderr)
        return chunks

def generate_f5_audio(f5tts, text, ref_audio, ref_text, f5_config=None):
    """Generate audio using F5-TTS - ONLY use config parameters that exist"""
    try:
        print(f"STATUS: Generating F5 audio ({len(text)} chars)", file=sys.stderr)
        print(f"STATUS: Using reference: {Path(ref_audio).name}", file=sys.stderr)
        
        # F5-TTS requires both ref_file and ref_text parameters
        infer_params = {
            'ref_file': ref_audio,
            'ref_text': ref_text,  # Empty string = auto-transcribe
            'gen_text': text
        }
        
        # Add parameters that exist in config ONLY - NO defaults
        if f5_config:
            # List of possible F5 parameters - only add if present in config
            possible_params = [
                'speed', 'cross_fade_duration', 'sway_sampling_coef', 
                'cfg_strength', 'nfe_step', 'seed', 'fix_duration', 'remove_silence'
            ]
            
            added_params = []
            for param in possible_params:
                if param in f5_config and f5_config[param] is not None:
                    infer_params[param] = f5_config[param]
                    added_params.append(f"{param}={f5_config[param]}")
            
            if added_params:
                print(f"STATUS: F5 parameters from config: {', '.join(added_params)}", file=sys.stderr)
            else:
                print(f"STATUS: Using F5 library defaults (no optional params in config)", file=sys.stderr)
        
        print(f"STATUS: F5 infer_params: {infer_params}", file=sys.stderr)
        
        # Generate audio
        result = f5tts.infer(**infer_params)
        
        # Handle different return formats
        if isinstance(result, (tuple, list)) and len(result) >= 2:
            audio, sr = result[0], result[1]
        else:
            audio, sr = result, 24000
        
        return audio, sr
        
    except Exception as e:
        print(f"ERROR: F5 generation failed: {e}", file=sys.stderr)
        import traceback
        print(f"TRACEBACK: {traceback.format_exc()}", file=sys.stderr)
        return None, None

def save_f5_audio(audio_data, sample_rate, output_path):
    """Save F5 audio to file"""
    if audio_data is None:
        return False
    
    try:
        if isinstance(audio_data, torch.Tensor):
            waveform = audio_data
        else:
            waveform = torch.tensor(audio_data, dtype=torch.float32)
        
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Normalize to avoid clipping
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val * 0.95
        
        # Save using torchaudio
        torchaudio.save(str(output_path), waveform.cpu(), sample_rate)
        print(f"STATUS: Saved F5 audio: {output_path} (max_val: {max_val:.3f})", file=sys.stderr)
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to save F5 audio: {e}", file=sys.stderr)
        return False

def process_f5_single(text, output_dir, f5_config, ref_audio, ref_text):
    """Process entire text as single batch - ONLY use config parameters"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load F5 model
    print(f"STATUS: Loading F5 model...", file=sys.stderr)
    try:
        f5tts = F5TTS()
        print("STATUS: F5 model loaded successfully", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: Failed to load F5 model: {e}", file=sys.stderr)
        return []
    
    # Preprocess text according to config
    processed_text = preprocess_text_for_f5(text, f5_config)
    
    output_file = output_dir / "complete_f5.wav"
    
    start_time = time.time()
    audio_data, sample_rate = generate_f5_audio(f5tts, processed_text, ref_audio, ref_text, f5_config)
    generation_time = time.time() - start_time
    
    if audio_data is not None:
        if save_f5_audio(audio_data, sample_rate, output_file):
            print(f"STATUS: Complete audio generated in {generation_time:.1f}s", file=sys.stderr)
            return [str(output_file)]
        else:
            print(f"ERROR: Failed to save complete audio", file=sys.stderr)
    else:
        print(f"ERROR: Failed to generate audio", file=sys.stderr)
    
    return []

def process_f5_chunked(text, output_dir, f5_config, ref_audio, ref_text):
    """Process text in chunks - ONLY use config parameters"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load F5 model
    print(f"STATUS: Loading F5 model...", file=sys.stderr)
    try:
        f5tts = F5TTS()
        print("STATUS: F5 model loaded successfully", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: Failed to load F5 model: {e}", file=sys.stderr)
        return []
    
    # Create chunks according to config
    chunks = chunk_text_for_f5(text, f5_config)
    generated_files = []
    
    for i, chunk_text in enumerate(chunks):
        chunk_num = i + 1
        output_file = output_dir / f"chunk_{chunk_num:03d}_f5.wav"
        
        print(f"STATUS: Processing chunk {chunk_num}/{len(chunks)} ({len(chunk_text)} chars)", file=sys.stderr)
        
        start_time = time.time()
        audio_data, sample_rate = generate_f5_audio(f5tts, chunk_text, ref_audio, ref_text, f5_config)
        generation_time = time.time() - start_time
        
        if audio_data is not None:
            if save_f5_audio(audio_data, sample_rate, output_file):
                generated_files.append(str(output_file))
                print(f"STATUS: Chunk {chunk_num} completed in {generation_time:.1f}s", file=sys.stderr)
            else:
                print(f"ERROR: Failed to save chunk {chunk_num}", file=sys.stderr)
        else:
            print(f"ERROR: Failed to generate audio for chunk {chunk_num}", file=sys.stderr)
    
    print(f"STATUS: F5 chunked processing completed: {len(generated_files)}/{len(chunks)} files", file=sys.stderr)
    return generated_files

def register_f5_engine():
    """Register F5 engine with the registry"""
    from engine_registry import register_engine
    
    register_engine(
        name='f5',
        processor_func=process_f5_text_file,
        default_config=get_f5_default_config()
    )