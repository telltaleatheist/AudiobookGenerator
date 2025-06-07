#!/usr/bin/env python3
"""
OpenAI Engine - OpenAI TTS processor with voice selection and quality control
INTEGRATED: Uses dynamic parameter loading from engine registry
Supports all OpenAI TTS models and voices with comprehensive error handling
"""

import sys
import os
import time
import requests # type: ignore
from pathlib import Path

# Import dynamic utilities from engine registry
from engines.base_engine import (
    extract_engine_config, 
    filter_params_for_function,
    create_generation_params
)

# OpenAI imports
try:
    import openai # type: ignore
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("ERROR: OpenAI library not available. Install with: pip install openai", file=sys.stderr)

def chunk_text_for_openai(text, max_chars=4000):
    """Split text into chunks optimized for OpenAI TTS"""
    # OpenAI TTS can handle very large chunks efficiently
    import re
    
    # If text is short enough, return as single chunk
    if len(text) <= max_chars:
        return [text]
    
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
    
    # Add remaining text
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def validate_openai_config(openai_config):
    """Validate OpenAI configuration and check API key"""
    # Check API key
    api_key = openai_config.get('api_key') or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OpenAI API key not found", file=sys.stderr)
        print("💡 Set OPENAI_API_KEY environment variable or add 'api_key' to config", file=sys.stderr)
        return False
    
    # Validate model
    valid_models = ['tts-1', 'tts-1-hd', 'gpt-4o-mini-tts']
    model = openai_config.get('model', 'tts-1')
    if model not in valid_models:
        print(f"WARNING: Unknown model '{model}'. Valid models: {', '.join(valid_models)}", file=sys.stderr)
    
    # Validate voice
    valid_voices = ['echo', 'fable', 'onyx', 'nova', 'shimmer', 'alloy']
    voice = openai_config.get('voice', 'onyx')
    if voice not in valid_voices:
        print(f"WARNING: Unknown voice '{voice}'. Valid voices: {', '.join(valid_voices)}", file=sys.stderr)
    
    # Validate speed
    speed = openai_config.get('speed', 1.0)
    if not (0.25 <= speed <= 4.0):
        print(f"WARNING: Speed {speed} outside valid range (0.25-4.0)", file=sys.stderr)
    
    return True

def setup_openai_client(openai_config):
    """Setup OpenAI client with API key"""
    api_key = openai_config.get('api_key') or os.getenv("OPENAI_API_KEY")
    
    if api_key:
        openai.api_key = api_key
        # For newer versions of openai library
        if hasattr(openai, 'OpenAI'):
            client = openai.OpenAI(api_key=api_key)
            return client
        else:
            return openai  # Older version, use module directly
    else:
        raise ValueError("OpenAI API key not found")

def generate_openai_audio_dynamic(client, text, openai_config):
    """Generate audio using OpenAI TTS with dynamic configuration support"""
    try:
        print(f"STATUS: Generating audio with OpenAI TTS", file=sys.stderr)
        print(f"STATUS: Text length: {len(text)} characters", file=sys.stderr)
        
        # Build generation parameters
        generation_params = {
            'model': openai_config.get('model', 'tts-1'),
            'voice': openai_config.get('voice', 'onyx'),
            'input': text
        }
        
        # Add optional parameters if present
        if 'speed' in openai_config:
            generation_params['speed'] = openai_config['speed']
        
        if 'response_format' in openai_config:
            generation_params['response_format'] = openai_config['response_format']
        
        print(f"STATUS: Model: {generation_params['model']}", file=sys.stderr)
        print(f"STATUS: Voice: {generation_params['voice']}", file=sys.stderr)
        if 'speed' in generation_params:
            print(f"STATUS: Speed: {generation_params['speed']}x", file=sys.stderr)
        
        # Generate audio
        if hasattr(client, 'audio') and hasattr(client.audio, 'speech'):
            # Newer client interface
            response = client.audio.speech.create(**generation_params)
        else:
            # Older interface
            response = client.audio.speech.create(**generation_params)
        
        # Get the audio content
        if hasattr(response, 'content'):
            audio_content = response.content
        else:
            audio_content = response
        
        print(f"STATUS: Audio generated successfully", file=sys.stderr)
        return audio_content
        
    except Exception as e:
        print(f"ERROR: OpenAI TTS generation failed: {e}", file=sys.stderr)
        
        # Provide helpful error messages
        if "authentication" in str(e).lower():
            print("💡 Check your OpenAI API key", file=sys.stderr)
        elif "quota" in str(e).lower():
            print("💡 Check your OpenAI account billing/quota", file=sys.stderr)
        elif "rate" in str(e).lower():
            print("💡 Rate limited - wait a moment before retrying", file=sys.stderr)
        
        return None

def save_openai_audio(audio_content, output_path, response_format='mp3'):
    """Save OpenAI audio content to file"""
    if audio_content is None:
        return False
    
    try:
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save based on format
        if response_format == 'mp3':
            # Save as MP3, then convert to WAV for consistency with other engines
            temp_mp3 = output_path.with_suffix('.mp3')
            with open(temp_mp3, 'wb') as f:
                f.write(audio_content)
            
            # Convert to WAV using ffmpeg if available
            try:
                import subprocess
                result = subprocess.run([
                    'ffmpeg', '-i', str(temp_mp3), '-y', str(output_path)
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    temp_mp3.unlink()  # Remove temp MP3
                    print(f"STATUS: Converted MP3 to WAV: {output_path.name}", file=sys.stderr)
                else:
                    # If conversion fails, keep MP3
                    output_path = temp_mp3
                    print(f"STATUS: Saved as MP3 (ffmpeg conversion failed): {output_path.name}", file=sys.stderr)
            except FileNotFoundError:
                # ffmpeg not available, keep MP3
                output_path = temp_mp3
                print(f"STATUS: Saved as MP3 (ffmpeg not available): {output_path.name}", file=sys.stderr)
                
        else:
            # Save in original format
            with open(output_path, 'wb') as f:
                f.write(audio_content)
            print(f"STATUS: Saved audio: {output_path.name}", file=sys.stderr)
        
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to save OpenAI audio: {e}", file=sys.stderr)
        return False

def estimate_openai_cost(text, model='tts-1'):
    """Estimate the cost for generating audio with OpenAI TTS"""
    char_count = len(text)
    
    # Pricing as of 2025 (per 1M characters)
    pricing = {
        'tts-1': 15.00,          # Standard TTS
        'tts-1-hd': 30.00,       # HD TTS
        'gpt-4o-mini-tts': 0.60  # Mini TTS (input cost)
    }
    
    cost_per_million = pricing.get(model, 15.00)
    estimated_cost = (char_count / 1_000_000) * cost_per_million
    
    return estimated_cost, char_count

def process_openai_chunks_with_retry(client, chunks, output_dir, openai_config):
    """Process chunks with retry logic and cost estimation"""
    generated_files = []
    retry_attempts = openai_config.get('retry_attempts', 3)
    retry_delay = openai_config.get('retry_delay', 1.0)
    model = openai_config.get('model', 'tts-1')
    response_format = openai_config.get('response_format', 'mp3')
    
    # Calculate total cost estimate
    total_chars = sum(len(chunk) for chunk in chunks)
    estimated_cost, _ = estimate_openai_cost(''.join(chunks), model)
    
    print(f"STATUS: Processing {len(chunks)} chunks ({total_chars:,} characters)", file=sys.stderr)
    print(f"STATUS: Estimated cost: ${estimated_cost:.4f} USD", file=sys.stderr)
    
    if estimated_cost > 1.0:  # Warn for costs over $1
        print(f"⚠️ WARNING: Estimated cost is ${estimated_cost:.2f}", file=sys.stderr)
    
    for i, chunk_text in enumerate(chunks):
        chunk_num = i + 1
        
        # Use appropriate extension
        if response_format == 'mp3':
            output_file = output_dir / f"chunk_{chunk_num:03d}_openai.wav"  # Will convert to WAV
        else:
            output_file = output_dir / f"chunk_{chunk_num:03d}_openai.{response_format}"
        
        chunk_cost, chunk_chars = estimate_openai_cost(chunk_text, model)
        print(f"STATUS: Processing chunk {chunk_num}/{len(chunks)} ({chunk_chars} chars, ${chunk_cost:.4f})", file=sys.stderr)
        
        # Retry logic
        success = False
        for attempt in range(retry_attempts):
            try:
                start_time = time.time()
                
                # Generate audio
                audio_content = generate_openai_audio_dynamic(client, chunk_text, openai_config)
                
                if audio_content is None:
                    if attempt < retry_attempts - 1:
                        print(f"WARNING: Attempt {attempt + 1} failed, retrying in {retry_delay}s", file=sys.stderr)
                        time.sleep(retry_delay)
                        continue
                    else:
                        print(f"ERROR: All {retry_attempts} attempts failed for chunk {chunk_num}", file=sys.stderr)
                        if not openai_config.get('ignore_errors', False):
                            break
                        continue
                
                generation_time = time.time() - start_time
                
                # Save audio
                if save_openai_audio(audio_content, output_file, response_format):
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
        
        if not success and not openai_config.get('skip_failed_chunks', False):
            print(f"ERROR: Critical failure on chunk {chunk_num}", file=sys.stderr)
            break
    
    return generated_files

def process_openai_text_file(text_file, output_dir, config, paths):
    """Main OpenAI engine processor with dynamic configuration detection"""
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI library not available. Install with: pip install openai")
    
    # Extract ALL OpenAI config parameters dynamically
    openai_config = extract_engine_config(config, 'openai', verbose=True)
    
    print(f"STATUS: Starting OpenAI TTS processing (dynamic mode)", file=sys.stderr)
    print(f"STATUS: Model: {openai_config.get('model', 'tts-1')}", file=sys.stderr)
    print(f"STATUS: Voice: {openai_config.get('voice', 'onyx')}", file=sys.stderr)
    
    # Display configured parameters
    active_params = {k: v for k, v in openai_config.items() if v is not None and k != 'api_key'}
    if active_params:
        print(f"STATUS: Active parameters: {active_params}", file=sys.stderr)
    
    # Validate configuration
    if not validate_openai_config(openai_config):
        return []
    
    # Setup OpenAI client
    try:
        client = setup_openai_client(openai_config)
        print(f"STATUS: OpenAI client initialized successfully", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: Failed to initialize OpenAI client: {e}", file=sys.stderr)
        return []
    
    # Read clean text
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    
    # Chunk text for OpenAI TTS
    chunk_max_chars = openai_config.get('chunk_max_chars', 4000)
    chunks = chunk_text_for_openai(text, chunk_max_chars)
    print(f"STATUS: Created {len(chunks)} chunks for OpenAI TTS", file=sys.stderr)
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process chunks with retry logic and cost tracking
    generated_files = process_openai_chunks_with_retry(client, chunks, output_dir, openai_config)
    
    # Final statistics
    success_rate = len(generated_files) / len(chunks) * 100 if chunks else 0
    print(f"STATUS: OpenAI TTS processing completed: {len(generated_files)}/{len(chunks)} files generated ({success_rate:.1f}% success)", file=sys.stderr)
    
    if len(generated_files) == 0:
        print(f"ERROR: No audio files were generated successfully", file=sys.stderr)
    
    return generated_files

def register_openai_engine():
    """Register OpenAI engine with the registry"""
    from engines.base_engine import register_engine
    
    register_engine(
        name='openai',
        processor_func=process_openai_text_file
    )