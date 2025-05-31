#!/usr/bin/env python3
"""
Edge Engine - EdgeTTS processor (free tier)
FIXED: Proper async handling and error checking
"""

import sys
import re
import time
import asyncio
from pathlib import Path

# EdgeTTS imports
try:
    import edge_tts
    EDGE_AVAILABLE = True
except ImportError:
    EDGE_AVAILABLE = False
    print("ERROR: EdgeTTS not available. Install with: pip install edge-tts", file=sys.stderr)

def get_edge_default_config():
    """Get default EdgeTTS configuration"""
    return {
        'edge': {
            'voice': 'en-US-AriaNeural',
            'rate': '+0%',
            'pitch': '+0Hz',
            'volume': '+0%',
            'chunk_max_chars': 1000,
            'target_chars': 800,
            'delay': 1.5
        }
    }

def chunk_text_for_edge(text, max_chars=1000):
    """Split text into chunks optimized for EdgeTTS"""
    # EdgeTTS can handle much larger chunks than Bark
    # Split into paragraphs first, then sentences if needed
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

async def generate_edge_audio(text, voice_config):
    """Generate audio using EdgeTTS (free tier, no SSML)"""
    try:
        print(f"DEBUG: Generating audio for {len(text)} characters with voice {voice_config['voice']}", file=sys.stderr)
        
        # Validate text
        if not text or not text.strip():
            print(f"ERROR: Empty text provided to EdgeTTS", file=sys.stderr)
            return None
        
        # Clean text for EdgeTTS
        text = text.strip()
        
        # Create communicate object
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice_config['voice'],
            rate=voice_config['rate'],
            pitch=voice_config['pitch'],
            volume=voice_config['volume']
        )
        
        print(f"DEBUG: Created EdgeTTS communicate object", file=sys.stderr)
        
        # Collect audio data
        audio_data = b""
        chunk_count = 0
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
                chunk_count += 1
        
        print(f"DEBUG: Received {chunk_count} audio chunks, total size: {len(audio_data)} bytes", file=sys.stderr)
        
        if len(audio_data) == 0:
            print(f"ERROR: No audio data received from EdgeTTS", file=sys.stderr)
            return None
        
        return audio_data
        
    except Exception as e:
        print(f"ERROR: EdgeTTS generation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return None

def save_edge_audio(audio_data, output_path):
    """Save EdgeTTS audio data to file"""
    if not audio_data:
        print(f"ERROR: No audio data to save", file=sys.stderr)
        return False
        
    try:
        with open(output_path, 'wb') as f:
            f.write(audio_data)
        
        print(f"DEBUG: Saved {len(audio_data)} bytes to {output_path}", file=sys.stderr)
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to save audio to {output_path}: {e}", file=sys.stderr)
        return False

async def process_edge_chunks_async(chunks, output_dir, edge_config):
    """Process chunks with EdgeTTS asynchronously"""
    generated_files = []
    
    for i, chunk_text in enumerate(chunks):
        chunk_num = i + 1
        output_file = output_dir / f"chunk_{chunk_num:03d}_edge.wav"
        
        print(f"STATUS: Processing chunk {chunk_num}/{len(chunks)} ({len(chunk_text)} chars)", file=sys.stderr)
        
        try:
            start_time = time.time()
            
            # Generate audio
            audio_data = await generate_edge_audio(chunk_text, edge_config)
            
            if audio_data is None:
                print(f"ERROR: No audio generated for chunk {chunk_num}", file=sys.stderr)
                continue
            
            generation_time = time.time() - start_time
            
            # Save audio
            if save_edge_audio(audio_data, output_file):
                generated_files.append(str(output_file))
                print(f"STATUS: Chunk {chunk_num} completed in {generation_time:.1f}s", file=sys.stderr)
            else:
                print(f"ERROR: Failed to save chunk {chunk_num}", file=sys.stderr)
                continue
            
            # Add delay between chunks to avoid throttling (except for last chunk)
            if i < len(chunks) - 1:
                print(f"DEBUG: Waiting {edge_config['delay']}s before next chunk", file=sys.stderr)
                await asyncio.sleep(edge_config['delay'])
                
        except Exception as e:
            print(f"ERROR: Failed to process chunk {chunk_num}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            continue
    
    return generated_files

def test_edge_voice(voice_name):
    """Test if a voice is available in EdgeTTS"""
    async def _test():
        try:
            voices = await edge_tts.list_voices()
            available_voices = [v["Name"] for v in voices]
            print(f"DEBUG: Found {len(available_voices)} total voices", file=sys.stderr)
            
            # Check if the requested voice exists
            voice_exists = voice_name in available_voices
            print(f"DEBUG: Voice '{voice_name}' exists: {voice_exists}", file=sys.stderr)
            
            # Get English voices for fallback
            en_voices = [v for v in available_voices if v.startswith('en-')]
            print(f"DEBUG: Found {len(en_voices)} English voices", file=sys.stderr)
            
            return voice_exists, en_voices
        except Exception as e:
            print(f"ERROR: Could not list EdgeTTS voices: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return False, []
    
    return asyncio.run(_test())

def process_edge_text_file(text_file, output_dir, config, paths):
    """Main EdgeTTS engine processor"""
    if not EDGE_AVAILABLE:
        raise ImportError("EdgeTTS not available. Install with: pip install edge-tts")
    
    # Get EdgeTTS config
    edge_config = config['edge']
    
    print(f"STATUS: Starting EdgeTTS processing", file=sys.stderr)
    print(f"STATUS: Voice: {edge_config['voice']}")
    print(f"STATUS: Rate: {edge_config['rate']}, Pitch: {edge_config['pitch']}", file=sys.stderr)
    
    # Test voice availability
    print(f"DEBUG: Testing voice availability for {edge_config['voice']}", file=sys.stderr)
    voice_available, available_voices = test_edge_voice(edge_config['voice'])
    
    if not voice_available:
        print(f"WARNING: Voice {edge_config['voice']} not found", file=sys.stderr)
        if available_voices:
            print(f"DEBUG: Available English voices: {available_voices[:5]}", file=sys.stderr)
            # Use the requested voice anyway - EdgeTTS might still work
            print(f"STATUS: Proceeding with {edge_config['voice']} anyway", file=sys.stderr)
        else:
            print(f"ERROR: Could not retrieve voice list, but proceeding anyway", file=sys.stderr)
    else:
        print(f"DEBUG: Voice {edge_config['voice']} confirmed available", file=sys.stderr)
    
    # Read clean text (already processed in preprocessing)
    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        print(f"DEBUG: Read {len(text)} characters from {text_file}", file=sys.stderr)
        
        if not text:
            print(f"ERROR: Text file is empty", file=sys.stderr)
            return []
            
    except Exception as e:
        print(f"ERROR: Could not read text file {text_file}: {e}", file=sys.stderr)
        return []
    
    # Chunk text for EdgeTTS
    chunks = chunk_text_for_edge(text, edge_config['chunk_max_chars'])
    print(f"STATUS: Created {len(chunks)} chunks for EdgeTTS", file=sys.stderr)
    
    if not chunks:
        print(f"ERROR: No chunks created from text", file=sys.stderr)
        return []
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process chunks asynchronously
    try:
        generated_files = asyncio.run(process_edge_chunks_async(chunks, output_dir, edge_config))
    except Exception as e:
        print(f"ERROR: Async processing failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return []
    
    print(f"STATUS: EdgeTTS processing completed: {len(generated_files)}/{len(chunks)} files generated", file=sys.stderr)
    
    if len(generated_files) == 0:
        print(f"ERROR: No audio files were generated successfully", file=sys.stderr)
    
    return generated_files

def register_edge_engine():
    """Register EdgeTTS engine with the registry"""
    from engine_registry import register_engine
    
    register_engine(
        name='edge',
        processor_func=process_edge_text_file,
        default_config=get_edge_default_config()
    )