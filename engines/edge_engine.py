#!/usr/bin/env python3
"""
Edge Engine - Free EdgeTTS processor (command-line version)
UPDATED: Now uses dynamic parameter loading from engine registry
Focuses ONLY on free edge-tts features, not paid Azure Speech Services
"""

import sys
import re
import time
import asyncio
from pathlib import Path

# Import dynamic utilities from engine registry
from engines.base_engine import (
    extract_engine_config, 
    filter_params_for_function,
    create_generation_params
)

# EdgeTTS imports (free version)
try:
    import edge_tts
    EDGE_AVAILABLE = True
except ImportError:
    EDGE_AVAILABLE = False
    print("ERROR: EdgeTTS not available. Install with: pip install edge-tts", file=sys.stderr)

def chunk_text_for_edge(text, max_chars=1000):
    """Split text into chunks optimized for EdgeTTS (free version can handle large chunks)"""
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

def apply_edge_text_preprocessing(text, edge_config):
    """Apply EdgeTTS text preprocessing based on dynamic config"""
    if not edge_config.get('normalize_text', True):
        return text
    
    processed_text = text
    
    # Apply various preprocessing based on config
    if edge_config.get('expand_abbreviations', True):
        # Basic abbreviation expansion for better TTS
        abbreviations = {
            'Dr.': 'Doctor',
            'Mr.': 'Mister', 
            'Mrs.': 'Missus',
            'Ms.': 'Miss',
            'Prof.': 'Professor',
            'etc.': 'and so on',
            'e.g.': 'for example',
            'i.e.': 'that is',
            'vs.': 'versus',
            'Inc.': 'Incorporated',
            'Ltd.': 'Limited',
            'Corp.': 'Corporation',
        }
        for abbrev, expansion in abbreviations.items():
            processed_text = re.sub(r'\b' + re.escape(abbrev) + r'\b', expansion, processed_text, flags=re.IGNORECASE)
    
    # Number handling
    if edge_config.get('spell_out_numbers', False):
        # Simple single digit spelling (basic implementation)
        number_words = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
        }
        for digit, word in number_words.items():
            processed_text = re.sub(r'\b' + digit + r'\b', word, processed_text)
    
    # Clean up extra whitespace
    processed_text = re.sub(r'\s+', ' ', processed_text)
    
    return processed_text.strip()

async def generate_edge_audio_dynamic(text, edge_config):
    """Generate audio using free EdgeTTS with dynamic configuration"""
    try:
        print(f"STATUS: Generating audio for {len(text)} characters with EdgeTTS", file=sys.stderr)
        
        # Validate text
        if not text or not text.strip():
            print(f"ERROR: Empty text provided to EdgeTTS", file=sys.stderr)
            return None
        
        # Apply text preprocessing
        processed_text = apply_edge_text_preprocessing(text.strip(), edge_config)
        
        # Build EdgeTTS communicate parameters (only free version parameters)
        communicate_params = {
            'text': processed_text,
            'voice': edge_config.get('voice', 'en-US-AriaNeural')
        }
        
        # Add free EdgeTTS parameters (rate, pitch, volume only)
        free_edge_params = ['rate', 'pitch', 'volume']
        for param in free_edge_params:
            if param in edge_config and edge_config[param] is not None:
                communicate_params[param] = edge_config[param]
                print(f"STATUS: Using EdgeTTS {param}: {edge_config[param]}", file=sys.stderr)
        
        # Create communicate object with filtered parameters
        valid_params = filter_params_for_function(edge_tts.Communicate.__init__, communicate_params, verbose=True)
        
        # Remove 'self' parameter that's not needed for instantiation
        if 'self' in valid_params:
            del valid_params['self']
        
        communicate = edge_tts.Communicate(**valid_params)
        
        print(f"STATUS: Created EdgeTTS communicate object", file=sys.stderr)
        
        # Collect audio data
        audio_data = b""
        chunk_count = 0
        
        # Handle streaming if configured (free version supports this)
        if edge_config.get('streaming', False):
            print(f"STATUS: Using streaming generation", file=sys.stderr)
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
                chunk_count += 1
        
        print(f"STATUS: Received {chunk_count} audio chunks, total size: {len(audio_data)} bytes", file=sys.stderr)
        
        if len(audio_data) == 0:
            print(f"ERROR: No audio data received from EdgeTTS", file=sys.stderr)
            return None
        
        return audio_data
        
    except Exception as e:
        print(f"ERROR: EdgeTTS generation failed: {e}", file=sys.stderr)
        if edge_config.get('debug_output', False):
            import traceback
            traceback.print_exc(file=sys.stderr)
        return None

def save_edge_audio_dynamic(audio_data, output_path, edge_config):
    """Save EdgeTTS audio data with optional post-processing from dynamic config"""
    if not audio_data:
        print(f"ERROR: No audio data to save", file=sys.stderr)
        return False
        
    try:
        # Note: Free EdgeTTS returns MP3 data, so post-processing is limited
        # Most audio effects would require conversion to WAV first
        
        # Save the audio (EdgeTTS free version outputs MP3 by default)
        with open(output_path, 'wb') as f:
            f.write(audio_data)
        
        print(f"STATUS: Saved {len(audio_data)} bytes to {output_path}", file=sys.stderr)
        
        # Log if advanced audio processing was requested but can't be done
        if edge_config.get('gain') and edge_config['gain'] != 0.0:
            print(f"INFO: Gain adjustment requested but not supported in free EdgeTTS", file=sys.stderr)
        if edge_config.get('normalize_audio'):
            print(f"INFO: Audio normalization requested but not supported in free EdgeTTS", file=sys.stderr)
        
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to save audio to {output_path}: {e}", file=sys.stderr)
        return False

async def process_edge_chunks_async_dynamic(chunks, output_dir, edge_config):
    """Process chunks with EdgeTTS asynchronously with dynamic retry logic"""
    generated_files = []
    
    # Get retry settings from dynamic config
    retry_attempts = edge_config.get('retry_attempts', 1)
    retry_delay = edge_config.get('retry_delay', 2.0)
    delay_between_chunks = edge_config.get('delay', 1.5)
    
    for i, chunk_text in enumerate(chunks):
        chunk_num = i + 1
        output_file = output_dir / f"chunk_{chunk_num:03d}_edge.wav"
        
        print(f"STATUS: Processing chunk {chunk_num}/{len(chunks)} ({len(chunk_text)} chars)", file=sys.stderr)
        
        # Retry logic
        success = False
        for attempt in range(retry_attempts):
            try:
                start_time = time.time()
                
                # Generate audio with dynamic config
                audio_data = await generate_edge_audio_dynamic(chunk_text, edge_config)
                
                if audio_data is None:
                    if attempt < retry_attempts - 1:
                        print(f"WARNING: Attempt {attempt + 1} failed, retrying in {retry_delay}s", file=sys.stderr)
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        print(f"ERROR: All {retry_attempts} attempts failed for chunk {chunk_num}", file=sys.stderr)
                        if not edge_config.get('ignore_errors', False):
                            break
                        continue
                
                generation_time = time.time() - start_time
                
                # Save audio with dynamic config
                if save_edge_audio_dynamic(audio_data, output_file, edge_config):
                    generated_files.append(str(output_file))
                    print(f"STATUS: Chunk {chunk_num} completed in {generation_time:.1f}s", file=sys.stderr)
                    success = True
                    break
                else:
                    if attempt < retry_attempts - 1:
                        print(f"WARNING: Save failed, retrying...", file=sys.stderr)
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        print(f"ERROR: Failed to save chunk {chunk_num} after {retry_attempts} attempts", file=sys.stderr)
                
            except Exception as e:
                if attempt < retry_attempts - 1:
                    print(f"ERROR: Attempt {attempt + 1} failed: {e}, retrying...", file=sys.stderr)
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    print(f"ERROR: Failed to process chunk {chunk_num} after {retry_attempts} attempts: {e}", file=sys.stderr)
                    if edge_config.get('debug_output', False):
                        import traceback
                        traceback.print_exc(file=sys.stderr)
                    break
        
        if not success and not edge_config.get('skip_failed_chunks', False):
            print(f"ERROR: Critical failure on chunk {chunk_num}", file=sys.stderr)
            break
        
        # Add delay between chunks to avoid throttling (except for last chunk)
        if i < len(chunks) - 1:
            print(f"STATUS: Waiting {delay_between_chunks}s before next chunk", file=sys.stderr)
            await asyncio.sleep(delay_between_chunks)
    
    return generated_files

def test_edge_voice_dynamic(voice_name, edge_config):
    """Test if a voice is available in free EdgeTTS with enhanced error handling"""
    async def _test():
        try:
            voices = await edge_tts.list_voices()
            available_voices = [v["Name"] for v in voices]
            
            if edge_config.get('verbose', False):
                print(f"STATUS: Found {len(available_voices)} total voices", file=sys.stderr)
            
            # Check if the requested voice exists
            voice_exists = voice_name in available_voices
            
            if edge_config.get('verbose', False):
                print(f"STATUS: Voice '{voice_name}' exists: {voice_exists}", file=sys.stderr)
            
            # Get language-specific voices for fallback
            lang_prefix = voice_name.split('-')[0:2]  # e.g., ['en', 'US']
            if len(lang_prefix) >= 2:
                lang_pattern = f"{lang_prefix[0]}-{lang_prefix[1]}"
                lang_voices = [v for v in available_voices if v.startswith(lang_pattern)]
            else:
                lang_voices = [v for v in available_voices if v.startswith('en-')]
            
            if edge_config.get('verbose', False):
                print(f"STATUS: Found {len(lang_voices)} voices for language pattern", file=sys.stderr)
            
            return voice_exists, lang_voices, available_voices
            
        except Exception as e:
            print(f"ERROR: Could not list EdgeTTS voices: {e}", file=sys.stderr)
            if edge_config.get('debug_output', False):
                import traceback
                traceback.print_exc(file=sys.stderr)
            return False, [], []
    
    return asyncio.run(_test())

def process_edge_text_file(text_file, output_dir, config, paths):
    """Main EdgeTTS engine processor with dynamic configuration detection (FREE VERSION ONLY)"""
    if not EDGE_AVAILABLE:
        raise ImportError("EdgeTTS not available. Install with: pip install edge-tts")
    
    # Extract ALL EdgeTTS config parameters dynamically using registry utilities
    edge_config = extract_engine_config(config, 'edge', verbose=True)
    
    print(f"STATUS: Starting EdgeTTS processing (free version, dynamic mode)", file=sys.stderr)
    print(f"STATUS: Voice: {edge_config.get('voice', 'en-US-AriaNeural')}", file=sys.stderr)
    
    # Show free EdgeTTS parameters
    rate = edge_config.get('rate', '+0%')
    pitch = edge_config.get('pitch', '+0Hz')
    volume = edge_config.get('volume', '+0%')
    print(f"STATUS: Rate: {rate}, Pitch: {pitch}, Volume: {volume}", file=sys.stderr)
    
    # Display configured features (only those supported by free version)
    free_features = []
    if edge_config.get('streaming'):
        free_features.append("streaming")
    if edge_config.get('normalize_text'):
        free_features.append("text normalization")
    if edge_config.get('expand_abbreviations'):
        free_features.append("abbreviation expansion")
    if edge_config.get('retry_attempts', 1) > 1:
        free_features.append(f"retry x{edge_config['retry_attempts']}")
    
    if free_features:
        print(f"STATUS: Free EdgeTTS features: {', '.join(free_features)}", file=sys.stderr)
    
    # Warn about unsupported paid features if they're in config
    paid_features = ['style', 'role', 'prosody_rate', 'prosody_pitch', 'emotion', 'voice_fixer', 'neural_voice']
    found_paid_features = [f for f in paid_features if edge_config.get(f) is not None]
    if found_paid_features:
        print(f"WARNING: Paid Azure Speech features detected but not supported in free EdgeTTS: {', '.join(found_paid_features)}", file=sys.stderr)
        print(f"INFO: Free EdgeTTS supports: voice, rate, pitch, volume only", file=sys.stderr)
    
    # Test voice availability with enhanced checking
    voice_name = edge_config.get('voice', 'en-US-AriaNeural')
    print(f"STATUS: Validating voice availability", file=sys.stderr)
    voice_available, lang_voices, all_voices = test_edge_voice_dynamic(voice_name, edge_config)
    
    if not voice_available:
        print(f"WARNING: Voice {voice_name} not found", file=sys.stderr)
        
        # Check for fallback voice
        fallback_voice = edge_config.get('fallback_voice')
        if fallback_voice and fallback_voice in all_voices:
            print(f"STATUS: Using fallback voice: {fallback_voice}", file=sys.stderr)
            edge_config['voice'] = fallback_voice
        elif lang_voices:
            print(f"STATUS: Available voices for language: {lang_voices[:3]}", file=sys.stderr)
            print(f"STATUS: Proceeding with {voice_name} anyway", file=sys.stderr)
        else:
            print(f"ERROR: Could not find suitable voice, but proceeding anyway", file=sys.stderr)
    else:
        print(f"STATUS: Voice {voice_name} confirmed available", file=sys.stderr)
    
    # Read clean text (already processed in preprocessing)
    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        if edge_config.get('verbose', False):
            print(f"STATUS: Read {len(text)} characters from {text_file}", file=sys.stderr)
        
        if not text:
            print(f"ERROR: Text file is empty", file=sys.stderr)
            return []
            
    except Exception as e:
        print(f"ERROR: Could not read text file {text_file}: {e}", file=sys.stderr)
        return []
    
    # Chunk text for EdgeTTS
    chunk_max_chars = edge_config.get('chunk_max_chars', 1000)
    chunks = chunk_text_for_edge(text, chunk_max_chars)
    print(f"STATUS: Created {len(chunks)} chunks for EdgeTTS", file=sys.stderr)
    
    if not chunks:
        print(f"ERROR: No chunks created from text", file=sys.stderr)
        return []
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process chunks asynchronously with dynamic features
    try:
        generated_files = asyncio.run(process_edge_chunks_async_dynamic(chunks, output_dir, edge_config))
    except Exception as e:
        print(f"ERROR: Async processing failed: {e}", file=sys.stderr)
        if edge_config.get('debug_output', False):
            import traceback
            traceback.print_exc(file=sys.stderr)
        return []
    
    # Final statistics
    success_rate = len(generated_files) / len(chunks) * 100 if chunks else 0
    print(f"STATUS: EdgeTTS processing completed: {len(generated_files)}/{len(chunks)} files generated ({success_rate:.1f}% success)", file=sys.stderr)
    
    if len(generated_files) == 0:
        print(f"ERROR: No audio files were generated successfully", file=sys.stderr)
    
    return generated_files

def register_edge_engine():
    """Register EdgeTTS (free) engine with the registry"""
    from engines.base_engine import register_engine
    
    # NO DEFAULT CONFIG NEEDED - everything comes from JSON file
    register_engine(
        name='edge',
        processor_func=process_edge_text_file
    )