#!/usr/bin/env python3
"""
Audio Enhancer - FFmpeg-based audio leveling and cleaning tool
Professional audio enhancement without voice cloning
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path

def get_default_config():
    """Get default FFmpeg configuration for audio enhancement"""
    return {
        "audio": {
            "speed_factor": 1.0,
            "clean_silence": True,
            "silence_threshold": -40.0,
            "silence_duration": 0.6,
            # FFmpeg Enhancement Options
            "loudness_target": -16.0,      # LUFS target for loudness normalization
            "true_peak_limit": -1.5,       # dBTP true peak limit
            "loudness_range": 11.0,        # LU loudness range target
            "highpass_frequency": 80,      # Hz - removes low frequency noise
            "noise_reduction": True,       # Enable noise reduction
            "noise_floor": -25,            # dB - noise reduction threshold
            "dynamic_compression": True,   # Enable dynamic range compression
            "limiter_enabled": True,       # Enable audio limiting
            "limiter_threshold": 0.95,     # Limiter threshold (0.0-1.0)
            "preserve_format": True,       # Keep original file format
            "force_format": None,          # Override format (wav, mp3, flac, mp4, mov, etc.)
            "sample_rate": 44100,          # Output sample rate
            "bit_depth": 16                # Output bit depth for WAV
        }
    }

def create_config_file():
    """Create a config.json file in the script directory"""
    script_dir = Path(__file__).parent
    config_path = script_dir / "config.json"
    
    config = get_default_config()
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Created config file: {config_path}")
        print(f"📝 You can edit this file to customize audio enhancement settings")
        print(f"💡 Copy this config.json to any folder where you want to process audio")
        print(f"\n🎛️ Key settings you can adjust:")
        print(f"   loudness_target: Target loudness in LUFS (-23 to -16)")
        print(f"   highpass_frequency: Remove low frequencies below this (Hz)")
        print(f"   noise_floor: Noise reduction threshold (dB)")
        print(f"   clean_silence: Remove long pauses (true/false)")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create config file: {e}")
        return False

def load_config(directory):
    """Load config.json from the specified directory"""
    config_path = Path(directory) / "config.json"
    
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"📄 Using config from: {config_path}")
            return config
        except Exception as e:
            print(f"⚠️ Error reading config file {config_path}: {e}")
            print(f"📄 Using default settings")
            return get_default_config()
    else:
        print(f"📄 No config file found in {directory}, using defaults")
        return get_default_config()

def get_supported_extensions():
    """Get list of supported audio/video file extensions"""
    return {
        # Audio formats
        '.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a', '.wma',
        # Video formats (audio will be extracted)
        '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'
    }

def find_audio_video_files(directory):
    """Find all audio/video files in a directory"""
    directory = Path(directory)
    supported_extensions = get_supported_extensions()
    
    files = []
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            files.append(file_path)
    
    return sorted(files)

def run_ffmpeg_enhancement(input_file, output_dir, audio_config):
    """Run FFmpeg-based audio enhancement"""
    input_path = Path(input_file)
    output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine output format
    if audio_config.get('force_format'):
        # Use forced format if specified
        output_format = audio_config['force_format']
        output_file = output_dir / f"{input_path.stem}.{output_format}"
    elif audio_config.get('preserve_format', True):
        # Keep original format (default behavior)
        output_format = input_path.suffix[1:].lower()  # Remove the dot
        output_file = output_dir / input_path.name
    else:
        # Fallback to wav
        output_format = 'wav'
        output_file = output_dir / f"{input_path.stem}.wav"
    
    print(f"🎧 Enhancing: {input_path.name}")
    print(f"💾 Output: {output_file.name} ({output_format.upper()})")
    
    # Build FFmpeg filter chain
    filters = []
    
    # 1. High-pass filter to remove low-frequency noise
    if audio_config.get('highpass_frequency', 80) > 0:
        freq = audio_config.get('highpass_frequency', 80)
        filters.append(f"highpass=f={freq}")
        print(f"   🔊 High-pass filter: {freq}Hz")
    
    # 2. Noise reduction
    if audio_config.get('noise_reduction', True):
        noise_floor = audio_config.get('noise_floor', -25)
        filters.append(f"afftdn=nf={noise_floor}")
        print(f"   🔇 Noise reduction: {noise_floor}dB")
    
    # 3. Dynamic range compression
    if audio_config.get('dynamic_compression', True):
        filters.append("compand=0.1,0.1:1,1:-80/-80|-62/-50|-25/-15|-12/-8:6:0:0.05:0.1")
        print(f"   📊 Dynamic compression enabled")
    
    # 4. Remove silence at start/end
    if audio_config.get('clean_silence', True):
        threshold = audio_config.get('silence_threshold', -40)
        duration = audio_config.get('silence_duration', 0.6)
        filters.append(f"silenceremove=start_periods=1:start_silence={duration}:start_threshold={threshold}dB:stop_periods=1:stop_silence={duration}:stop_threshold={threshold}dB")
        print(f"   ✂️ Silence removal: {threshold}dB, {duration}s")
    
    # 5. Audio limiting
    if audio_config.get('limiter_enabled', True):
        limit = audio_config.get('limiter_threshold', 0.95)
        filters.append(f"alimiter=limit={limit}:attack=5:release=50")
        print(f"   🚫 Audio limiter: {limit}")
    
    # 6. Loudness normalization (CRITICAL for consistent levels)
    loudness_target = audio_config.get('loudness_target', -16.0)
    true_peak = audio_config.get('true_peak_limit', -1.5)
    lra = audio_config.get('loudness_range', 11.0)
    filters.append(f"loudnorm=I={loudness_target}:TP={true_peak}:LRA={lra}")
    print(f"   📏 Loudness normalization: {loudness_target} LUFS")
    
    # 7. Speed adjustment if needed
    speed_factor = audio_config.get('speed_factor', 1.0)
    if speed_factor != 1.0:
        filters.append(f"atempo={speed_factor}")
        print(f"   ⏩ Speed adjustment: {speed_factor}x")
    
    # Combine all filters
    filter_complex = ",".join(filters)
    
    # Build FFmpeg command
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-af", filter_complex
    ]
    
    # Add output format options based on file type
    sample_rate = audio_config.get('sample_rate', 44100)
    
    if output_format.lower() in ['wav']:
        # Audio-only WAV format
        bit_depth = audio_config.get('bit_depth', 16)
        if bit_depth == 16:
            cmd.extend(["-c:a", "pcm_s16le"])
        elif bit_depth == 24:
            cmd.extend(["-c:a", "pcm_s24le"])
        else:
            cmd.extend(["-c:a", "pcm_s16le"])
        cmd.extend(["-ar", str(sample_rate)])
        
    elif output_format.lower() in ['mp3']:
        # Audio-only MP3 format
        cmd.extend(["-c:a", "libmp3lame", "-b:a", "320k"])
        cmd.extend(["-ar", str(sample_rate)])
        
    elif output_format.lower() in ['flac']:
        # Audio-only FLAC format
        cmd.extend(["-c:a", "flac"])
        cmd.extend(["-ar", str(sample_rate)])
        
    elif output_format.lower() in ['mp4', 'm4v', 'm4a']:
        # Video formats - copy video stream, enhance audio
        cmd.extend(["-c:v", "copy"])  # Copy video without re-encoding
        cmd.extend(["-c:a", "aac", "-b:a", "256k"])  # High-quality AAC audio
        cmd.extend(["-ar", str(sample_rate)])
        
    elif output_format.lower() in ['mov']:
        # MOV format - copy video stream, enhance audio
        cmd.extend(["-c:v", "copy"])  # Copy video without re-encoding
        cmd.extend(["-c:a", "aac", "-b:a", "256k"])  # High-quality AAC audio
        cmd.extend(["-ar", str(sample_rate)])
        
    elif output_format.lower() in ['avi']:
        # AVI format - copy video stream, enhance audio
        cmd.extend(["-c:v", "copy"])  # Copy video without re-encoding
        cmd.extend(["-c:a", "mp3", "-b:a", "320k"])  # MP3 audio for AVI compatibility
        cmd.extend(["-ar", str(sample_rate)])
        
    elif output_format.lower() in ['mkv', 'webm']:
        # MKV/WebM format - copy video stream, enhance audio
        cmd.extend(["-c:v", "copy"])  # Copy video without re-encoding
        cmd.extend(["-c:a", "libvorbis", "-b:a", "256k"])  # Vorbis audio
        cmd.extend(["-ar", str(sample_rate)])
        
    else:
        # Unknown format - try to preserve original codec
        cmd.extend(["-c:v", "copy"])  # Copy video if present
        cmd.extend(["-c:a", "aac", "-b:a", "256k"])  # Use AAC as safe fallback
        cmd.extend(["-ar", str(sample_rate)])
        print(f"   ⚠️ Unknown format '{output_format}', using AAC audio")
    
    cmd.append(str(output_file))
    
    print(f"🔧 Processing with {len(filters)} audio filters...")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Enhancement successful")
        return True, output_file
        
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg failed:")
        print(f"   Command: {' '.join(cmd[:5])}...")
        print(f"   Error: {e.stderr[:300] if e.stderr else str(e)}")
        
        # Try simplified version
        return run_simple_enhancement(input_file, output_dir, audio_config)
        
    except FileNotFoundError:
        print(f"❌ FFmpeg not found. Please install FFmpeg")
        return False, None

def run_simple_enhancement(input_file, output_dir, audio_config):
    """Run simplified FFmpeg enhancement if advanced version fails"""
    input_path = Path(input_file)
    output_dir = Path(output_dir)
    
    # Determine output format (same logic as main function)
    if audio_config.get('force_format'):
        output_format = audio_config['force_format']
        output_file = output_dir / f"{input_path.stem}_simple.{output_format}"
    elif audio_config.get('preserve_format', True):
        output_format = input_path.suffix[1:].lower()
        output_file = output_dir / f"{input_path.stem}_simple{input_path.suffix}"
    else:
        output_format = 'wav'
        output_file = output_dir / f"{input_path.stem}_simple.wav"
    
    print(f"🔄 Trying simplified enhancement...")
    
    # Minimal, most compatible filters
    filters = []
    
    # Basic loudness normalization
    loudness_target = audio_config.get('loudness_target', -16.0)
    filters.append(f"loudnorm=I={loudness_target}")
    
    # Basic high-pass filter
    if audio_config.get('highpass_frequency', 80) > 0:
        freq = audio_config.get('highpass_frequency', 80)
        filters.append(f"highpass=f={freq}")
    
    filter_complex = ",".join(filters)
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-af", filter_complex
    ]
    
    # Simple codec selection based on format
    if output_format.lower() in ['mp4', 'mov', 'm4v']:
        cmd.extend(["-c:v", "copy", "-c:a", "aac", "-b:a", "256k"])
    elif output_format.lower() in ['avi']:
        cmd.extend(["-c:v", "copy", "-c:a", "mp3", "-b:a", "320k"])
    else:
        cmd.extend(["-c:a", "pcm_s16le"])
    
    cmd.extend(["-ar", "44100", str(output_file)])
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Simplified enhancement successful")
        return True, output_file
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Simplified enhancement also failed")
        
        # Last resort: just copy with format conversion
        return copy_with_conversion(input_file, output_dir, audio_config)
        
    except FileNotFoundError:
        print(f"❌ FFmpeg not found")
        return False, None

def copy_with_conversion(input_file, output_dir, audio_config):
    """Last resort: convert format without enhancement"""
    input_path = Path(input_file)
    output_dir = Path(output_dir)
    
    # Determine output format (same logic)
    if audio_config.get('force_format'):
        output_format = audio_config['force_format']
        output_file = output_dir / f"{input_path.stem}_converted.{output_format}"
    elif audio_config.get('preserve_format', True):
        output_format = input_path.suffix[1:].lower()
        output_file = output_dir / f"{input_path.stem}_converted{input_path.suffix}"
    else:
        output_format = 'wav'
        output_file = output_dir / f"{input_path.stem}_converted.wav"
    
    print(f"🔄 Converting without enhancement...")
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path)
    ]
    
    # Basic codec selection
    if output_format.lower() in ['mp4', 'mov', 'm4v']:
        cmd.extend(["-c:v", "copy", "-c:a", "aac"])
    elif output_format.lower() in ['avi']:
        cmd.extend(["-c:v", "copy", "-c:a", "mp3"])
    else:
        cmd.extend(["-c:a", "pcm_s16le"])
    
    cmd.extend(["-ar", "44100", str(output_file)])
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"⚠️ File converted without enhancement")
        return True, output_file
        
    except Exception as e:
        print(f"❌ All processing failed: {e}")
        return False, None

def process_single_file(input_file, config):
    """Process a single audio/video file"""
    input_path = Path(input_file)
    
    # Create cleaned directory in the same folder as the input file
    cleaned_dir = input_path.parent / "cleaned"
    cleaned_dir.mkdir(exist_ok=True)
    
    print(f"📁 Processing: {input_path}")
    print(f"📁 Output directory: {cleaned_dir}")
    
    # Process the file
    success, output_file = run_ffmpeg_enhancement(input_path, cleaned_dir, config['audio'])
    
    if success and output_file:
        print(f"✅ File enhanced successfully: {output_file}")
        return True
    else:
        print(f"❌ Enhancement failed")
        return False

def process_directory(directory, config):
    """Process all audio/video files in a directory"""
    directory = Path(directory)
    
    # Find all supported files
    files = find_audio_video_files(directory)
    
    if not files:
        print(f"❌ No audio/video files found in {directory}")
        return False
    
    print(f"📁 Found {len(files)} files to process")
    
    # Create cleaned directory
    cleaned_dir = directory / "cleaned"
    cleaned_dir.mkdir(exist_ok=True)
    
    successful = 0
    failed = 0
    
    for file_path in files:
        print(f"\n📼 Processing {successful + failed + 1}/{len(files)}: {file_path.name}")
        
        try:
            success, output_file = run_ffmpeg_enhancement(file_path, cleaned_dir, config['audio'])
            
            if success and output_file:
                print(f"✅ Enhanced: {file_path.name} → {output_file.name}")
                successful += 1
            else:
                print(f"❌ Failed: {file_path.name}")
                failed += 1
                    
        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")
            failed += 1
    
    print(f"\n🎉 Processing complete!")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Enhanced files saved to: {cleaned_dir}")
    
    return successful > 0

def get_user_input():
    """Get file or directory path from user input"""
    while True:
        path_input = input("\n📂 Enter file or directory path: ").strip()
        
        if not path_input:
            print("❌ Please enter a valid path")
            continue
        
        # Remove quotes if present
        path_input = path_input.strip('"\'')
        
        path = Path(path_input)
        
        if not path.exists():
            print(f"❌ Path does not exist: {path}")
            continue
        
        return path

def main():
    parser = argparse.ArgumentParser(
        description="Audio Enhancement Tool using FFmpeg",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create config file
  python audio_enhancer.py --create-config
  
  # Process single file
  python audio_enhancer.py --input audio.wav
  
  # Process directory
  python audio_enhancer.py --input /path/to/audio/folder
  
  # Interactive mode (no arguments)
  python audio_enhancer.py

Audio Enhancement Features:
  • Loudness normalization for consistent volume
  • Noise reduction and high-pass filtering
  • Dynamic range compression
  • Silence removal and audio limiting
  • Professional broadcast-standard processing
        """
    )
    
    parser.add_argument("--input", "-i", help="Input file or directory")
    parser.add_argument("--create-config", action="store_true", 
                       help="Create default config.json file")
    
    args = parser.parse_args()
    
    # Handle config creation
    if args.create_config:
        return 0 if create_config_file() else 1
    
    # Get input path
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"❌ Input path does not exist: {input_path}")
            return 1
    else:
        # Interactive mode
        print("🎵 Audio Enhancement Tool")
        print("=" * 50)
        input_path = get_user_input()
    
    # Determine config directory
    if input_path.is_file():
        config_dir = input_path.parent
    else:
        config_dir = input_path
    
    # Load configuration
    config = load_config(config_dir)
    
    # Display settings
    audio_config = config['audio']
    print(f"\n🎛️ Enhancement Settings:")
    print(f"   Loudness target: {audio_config.get('loudness_target', -16)} LUFS")
    print(f"   High-pass filter: {audio_config.get('highpass_frequency', 80)} Hz")
    print(f"   Format handling: {'Preserve original' if audio_config.get('preserve_format', True) else 'Force ' + str(audio_config.get('force_format', 'wav')).upper()}")
    print(f"   Sample rate: {audio_config.get('sample_rate', 44100)} Hz")
    
    # Process files
    try:
        if input_path.is_file():
            # Check if it's a supported file type
            if input_path.suffix.lower() not in get_supported_extensions():
                print(f"❌ Unsupported file type: {input_path.suffix}")
                print(f"📝 Supported formats: {', '.join(sorted(get_supported_extensions()))}")
                return 1
            
            success = process_single_file(input_path, config)
        else:
            success = process_directory(input_path, config)
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())