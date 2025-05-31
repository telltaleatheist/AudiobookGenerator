#!/usr/bin/env python3
"""
Edge Engine - EdgeTTS processor with SSML support
Handles larger chunks and pronunciation fixes via SSML
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
            'delay': 1.5,
            'use_ssml': True
        }
    }

def apply_edge_pronunciation_fixes(text, use_ssml=True):
    """Apply EdgeTTS-specific pronunciation fixes"""
    if not use_ssml:
        return text, 0
    
    # EdgeTTS SSML pronunciation fixes using IPA phonemes
    ssml_fixes = {
        # Religious/philosophical terms
        "atheist": '<phoneme alphabet="ipa" ph="ˈeɪθiɪst">atheist</phoneme>',
        "atheists": '<phoneme alphabet="ipa" ph="ˈeɪθiɪsts">atheists</phoneme>',
        "atheism": '<phoneme alphabet="ipa" ph="ˈeɪθiˌɪzəm">atheism</phoneme>',
        
        # Common mispronunciations
        "colonel": '<phoneme alphabet="ipa" ph="ˈkɜrnəl">colonel</phoneme>',
        "hierarchy": '<phoneme alphabet="ipa" ph="ˈhaɪərˌɑrki">hierarchy</phoneme>',
        "epitome": '<phoneme alphabet="ipa" ph="ɪˈpɪtəmi">epitome</phoneme>',
        "hyperbole": '<phoneme alphabet="ipa" ph="haɪˈpɜrbəli">hyperbole</phoneme>',
        "cache": '<phoneme alphabet="ipa" ph="kæʃ">cache</phoneme>',
        "niche": '<phoneme alphabet="ipa" ph="niʃ">niche</phoneme>',
        "facade": '<phoneme alphabet="ipa" ph="fəˈsɑd">facade</phoneme>',
        "gauge": '<phoneme alphabet="ipa" ph="ɡeɪdʒ">gauge</phoneme>',
        
        # Historical terms
        "bourgeois": '<phoneme alphabet="ipa" ph="bʊrˈʒwɑ">bourgeois</phoneme>',
        "regime": '<phoneme alphabet="ipa" ph="reɪˈʒim">regime</phoneme>',
        "fascism": '<phoneme alphabet="ipa" ph="ˈfæʃˌɪzəm">fascism</phoneme>',
        "Nazi": '<phoneme alphabet="ipa" ph="ˈnɑtsi">Nazi</phoneme>',
        "Nazis": '<phoneme alphabet="ipa" ph="ˈnɑtsiz">Nazis</phoneme>',
        "Aryan": '<phoneme alphabet="ipa" ph="ˈɛriən">Aryan</phoneme>',
        
        # Geographic names
        "Worcester": '<phoneme alphabet="ipa" ph="ˈwʊstər">Worcester</phoneme>',
        "Leicester": '<phoneme alphabet="ipa" ph="ˈlɛstər">Leicester</phoneme>',
        "Arkansas": '<phoneme alphabet="ipa" ph="ˈɑrkənˌsɔ">Arkansas</phoneme>',
    }
    
    fixes_applied = 0
    for word, ssml_replacement in ssml_fixes.items():
        pattern = r'\b' + re.escape(word) + r'\b'
        matches = len(re.findall(pattern, text, flags=re.IGNORECASE))
        if matches > 0:
            text = re.sub(pattern, ssml_replacement, text, flags=re.IGNORECASE)
            fixes_applied += matches
    
    if fixes_applied > 0:
        print(f"STATUS: Applied {fixes_applied} SSML pronunciation fixes", file=sys.stderr)
    
    return text, fixes_applied

def add_ssml_breaks(text):
    """Add SSML break tags for natural pauses"""
    text = re.sub(r'([.!?])\s+', r'\1 <break time="500ms"/> ', text)
    text = re.sub(r'([;:])\s+', r'\1 <break time="250ms"/> ', text)
    text = re.sub(r'(,)\s+', r'\1 <break time="200ms"/> ', text)
    return text

def wrap_in_ssml(text, voice_config):
    """Wrap text in SSML with voice settings"""
    ssml = f'''<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
    <voice name="{voice_config['voice']}">
        <prosody rate="{voice_config['rate']}" pitch="{voice_config['pitch']}" volume="{voice_config['volume']}">
            {text}
        </prosody>
    </voice>
</speak>'''
    return ssml

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

async def generate_edge_audio(text, voice_config, use_ssml=False):
    """Generate audio using EdgeTTS"""
    if use_ssml:
        # Apply pronunciation fixes and wrap in SSML
        text, _ = apply_edge_pronunciation_fixes(text, use_ssml=True)
        text = add_ssml_breaks(text)
        ssml_text = wrap_in_ssml(text, voice_config)
        
        communicate = edge_tts.Communicate(ssml=ssml_text)
    else:
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice_config['voice'],
            rate=voice_config['rate'],
            pitch=voice_config['pitch'],
            volume=voice_config['volume']
        )
    
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    
    return audio_data

def save_edge_audio(audio_data, output_path):
    """Save EdgeTTS audio data to file"""
    with open(output_path, 'wb') as f:
        f.write(audio_data)

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
            audio_data = await generate_edge_audio(
                chunk_text, 
                edge_config, 
                use_ssml=edge_config['use_ssml']
            )
            
            generation_time = time.time() - start_time
            
            # Save audio
            save_edge_audio(audio_data, output_file)
            generated_files.append(str(output_file))
            
            print(f"STATUS: Chunk {chunk_num} completed in {generation_time:.1f}s", file=sys.stderr)
            
            # Add delay between chunks to avoid throttling (except for last chunk)
            if i < len(chunks) - 1:
                await asyncio.sleep(edge_config['delay'])
                
        except Exception as e:
            print(f"ERROR: Failed to process chunk {chunk_num}: {e}", file=sys.stderr)
            continue
    
    return generated_files

def process_edge_text_file(text_file, output_dir, config, paths):
    """Main EdgeTTS engine processor"""
    if not EDGE_AVAILABLE:
        raise ImportError("EdgeTTS not available. Install with: pip install edge-tts")
    
    # Get EdgeTTS config
    edge_config = config['edge']
    
    print(f"STATUS: Starting EdgeTTS processing", file=sys.stderr)
    print(f"STATUS: Voice: {edge_config['voice']}")
    print(f"STATUS: Rate: {edge_config['rate']}, Pitch: {edge_config['pitch']}", file=sys.stderr)
    print(f"STATUS: SSML: {'enabled' if edge_config['use_ssml'] else 'disabled'}", file=sys.stderr)
    
    # Read clean text
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    
    # Chunk text for EdgeTTS
    chunks = chunk_text_for_edge(text, edge_config['chunk_max_chars'])
    print(f"STATUS: Created {len(chunks)} chunks for EdgeTTS", file=sys.stderr)
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process chunks asynchronously
    generated_files = asyncio.run(process_edge_chunks_async(chunks, output_dir, edge_config))
    
    print(f"STATUS: EdgeTTS processing completed: {len(generated_files)}/{len(chunks)} files generated", file=sys.stderr)
    return generated_files

def register_edge_engine():
    """Register EdgeTTS engine with the registry"""
    from engine_registry import register_engine
    
    register_engine(
        name='edge',
        processor_func=process_edge_text_file,
        default_config=get_edge_default_config()
    )