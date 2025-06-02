#!/usr/bin/env python3
"""
Audio Processor - Handles RVC conversion and final audio cleanup
FIXED: Only use RVC parameters that are actually supported by the installed version
"""

import sys
import subprocess
import tempfile
import shutil
import os
from pathlib import Path

def get_audio_default_config():
    """Get default audio processing configuration matching your preferred settings"""
    return {
        'rvc': {
            'model': 'my_voice',  # Match your config
            'speed_factor': 0.8,  # Match your config
            'clean_silence': True,
            'silence_threshold': -40.0,
            'silence_duration': 0.5,  # Match your config
            # Core RVC configuration options - only the ones that are widely supported
            'n_semitones': -2,
            'f0_method': 'crepe',
            'index_rate': 0.4,
            'protect_rate': 0.5,  # Match your config
            'rms_mix_rate': 0.3,  # Match your config
            'hop_length': 64,
            'split_voice': True,
            'clean_voice': True,
            'clean_strength': 0.3,  # Match your config
            'autotune_voice': True,
            'autotune_strength': 0.2,  # Match your config
            # Removed optional parameters that may not be supported in all RVC versions:
            # 'filter_radius', 'resample_sr', 'envelope_mix', 'protect_voiceless'
        },
        'audio': {
            'silence_gap': 0.3
        }
    }

def combine_audio_files(audio_files, output_path, silence_gap=0.3):
    """Combine multiple audio files with silence gaps"""
    print(f"STATUS: Combining {len(audio_files)} audio files", file=sys.stderr)
    
    if not audio_files:
        raise ValueError("No audio files to combine")
    
    if len(audio_files) == 1:
        # Single file, just copy it
        shutil.copy2(audio_files[0], output_path)
        print(f"STATUS: Single file copied", file=sys.stderr)
        return True
    
    try:
        if silence_gap > 0:
            # Use sox if available, otherwise fallback to simple concat
            try:
                # Try using sox for simple concatenation with padding
                cmd = ["sox"] + [str(f) for f in audio_files] + [str(output_path), "pad", "0", str(silence_gap)]
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"STATUS: Audio files combined with sox", file=sys.stderr)
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Sox not available or failed, use ffmpeg with simpler approach
                pass
        
        # Fallback: simple concatenation without silence using ffmpeg demuxer
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for audio_file in audio_files:
                f.write(f"file '{Path(audio_file).absolute()}'\n")
            file_list = f.name
        
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0", 
            "-i", file_list,
            "-c", "copy",
            str(output_path)
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"STATUS: Audio files combined successfully", file=sys.stderr)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to combine audio files: {e}", file=sys.stderr)
        print(f"STDERR: {e.stderr}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print("ERROR: ffmpeg not found. Please install ffmpeg", file=sys.stderr)
        return False
    finally:
        # Clean up temp file
        try:
            Path(file_list).unlink()
        except:
            pass

def speed_up_audio(input_path, output_path, speed_factor=1.0):
    """Speed up audio by the specified factor while preserving pitch"""
    if speed_factor == 1.0:
        # No speed change needed
        shutil.copy2(input_path, output_path)
        return True
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-af", f"atempo={speed_factor}",
        str(output_path)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"STATUS: Audio sped up by {speed_factor}x", file=sys.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to speed up audio: {e}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print("ERROR: ffmpeg not found. Please install ffmpeg", file=sys.stderr)
        return False

def remove_long_silence(input_path, output_path, silence_threshold=-40, silence_duration=0.6):
    """Remove only long silences from middle/end of audio, preserve beginning timing"""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-af", f"silenceremove=stop_periods=-1:stop_silence={silence_duration}:stop_threshold={silence_threshold}dB",
        str(output_path)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"STATUS: Long silences removed", file=sys.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to remove silence: {e}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print("ERROR: ffmpeg not found. Please install ffmpeg", file=sys.stderr)
        return False

def run_rvc_conversion(input_wav, output_dir, model_name="Sigma Male Narrator", rvc_config=None):
    """Run RVC conversion with only the core supported configuration settings"""
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use provided config or fall back to defaults
    if not rvc_config:
        rvc_config = get_audio_default_config()['rvc']
    
    print(f"STATUS: Running RVC conversion with model '{model_name}'", file=sys.stderr)
    
    # Build command with ONLY the core configurable values that are widely supported
    cmd = [
        "urvc", "generate", "convert-voice",
        str(input_wav),
        str(output_dir),
        model_name,
        "--n-semitones", str(rvc_config.get('n_semitones', -2)),
        "--f0-method", rvc_config.get('f0_method', 'crepe'),
        "--index-rate", str(rvc_config.get('index_rate', 0.4)),
        "--protect-rate", str(rvc_config.get('protect_rate', 0.4)),
        "--rms-mix-rate", str(rvc_config.get('rms_mix_rate', 0.5)),
        "--hop-length", str(rvc_config.get('hop_length', 64))
    ]
    
    # Add boolean flags if enabled (these are widely supported)
    boolean_flags = {
        'split_voice': '--split-voice',
        'clean_voice': '--clean-voice',
        'autotune_voice': '--autotune-voice'
    }
    
    for config_key, cmd_flag in boolean_flags.items():
        if rvc_config.get(config_key, False):
            cmd.append(cmd_flag)
            print(f"STATUS: RVC {config_key} enabled", file=sys.stderr)
    
    # Add strength parameters for enabled features (widely supported)
    if rvc_config.get('clean_voice', False) and 'clean_strength' in rvc_config:
        cmd.extend(["--clean-strength", str(rvc_config['clean_strength'])])
        print(f"STATUS: Using RVC clean_strength: {rvc_config['clean_strength']}", file=sys.stderr)
    
    if rvc_config.get('autotune_voice', False) and 'autotune_strength' in rvc_config:
        cmd.extend(["--autotune-strength", str(rvc_config['autotune_strength'])])
        print(f"STATUS: Using RVC autotune_strength: {rvc_config['autotune_strength']}", file=sys.stderr)
    
    print(f"DEBUG: Full RVC command: {' '.join(cmd)}", file=sys.stderr)
    
    try:
        # Fix environment variables for RVC
        env = os.environ.copy()
        
        # Set PYTHONHASHSEED to a fixed value for reproducibility
        env['PYTHONHASHSEED'] = '42'
        
        # Ensure PATH includes conda environment if we're in one
        if 'CONDA_DEFAULT_ENV' in env:
            conda_path = os.path.dirname(sys.executable)
            if conda_path not in env.get('PATH', ''):
                env['PATH'] = conda_path + os.pathsep + env.get('PATH', '')
        
        # Run RVC with proper environment
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True,
            env=env,
            cwd=os.getcwd()  # Ensure we're in the right working directory
        )
        
        print("STATUS: RVC conversion completed successfully", file=sys.stderr)
        return True, result.stdout
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: RVC conversion failed (return code {e.returncode})", file=sys.stderr)
        print(f"STDERR: {e.stderr}", file=sys.stderr)
        
        # Additional debugging info
        print(f"DEBUG: Command: {' '.join(cmd)}", file=sys.stderr)
        print(f"DEBUG: Working directory: {os.getcwd()}", file=sys.stderr)
        print(f"DEBUG: Python executable: {sys.executable}", file=sys.stderr)
        
        return False, None
        
    except FileNotFoundError:
        print("ERROR: RVC command 'urvc' not found. Make sure RVC is installed and in PATH", file=sys.stderr)
        print(f"DEBUG: Current PATH: {os.environ.get('PATH', 'Not found')}", file=sys.stderr)
        return False, None
    
def find_rvc_output_file(output_dir, base_name=None):
    """Find the RVC output file (RVC typically adds suffixes to filenames)"""
    output_dir = Path(output_dir)
    
    # Common patterns RVC might use
    patterns = [
        "*.wav",  # Try all wav files first
        "converted_*.wav",
        "*_converted.wav",
        "*Voice_Converted*.wav"  # Pattern from your successful manual run
    ]
    
    if base_name:
        patterns = [
            f"{base_name}*.wav",
            f"*{base_name}*.wav"
        ] + patterns
    
    for pattern in patterns:
        matches = list(output_dir.glob(pattern))
        if matches:
            # Return the most recent file if multiple matches
            return max(matches, key=lambda p: p.stat().st_mtime)
    
    return None

def process_audio_through_rvc(input_file, output_file, config):
    """Process an audio file through RVC conversion with core configuration settings"""
    
    input_path = Path(input_file)
    output_path = Path(output_file)
    rvc_config = config['rvc']
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"STATUS: Processing {input_path.name} through RVC", file=sys.stderr)
    print(f"STATUS: RVC model: {rvc_config['model']}", file=sys.stderr)
    
    # Log core RVC settings being used (only the ones we're actually passing)
    print(f"STATUS: RVC settings:", file=sys.stderr)
    core_settings = [
        'speed_factor', 'clean_silence', 'silence_threshold', 'silence_duration',
        'n_semitones', 'f0_method', 'index_rate', 'protect_rate', 'rms_mix_rate', 
        'hop_length', 'split_voice', 'clean_voice', 'clean_strength', 
        'autotune_voice', 'autotune_strength'
    ]
    
    for key in core_settings:
        if key in rvc_config:
            print(f"  {key}: {rvc_config[key]}", file=sys.stderr)
    
    # Create temporary directory for RVC processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Run RVC conversion with core config only
        success, stdout = run_rvc_conversion(input_path, temp_path, rvc_config['model'], rvc_config)
        
        if not success:
            return False
        
        # Find RVC output file
        rvc_output = find_rvc_output_file(temp_path, input_path.stem)
        
        if not rvc_output:
            print("ERROR: Could not find RVC output file", file=sys.stderr)
            print(f"DEBUG: Contents of {temp_path}:", file=sys.stderr)
            for file in temp_path.glob("*"):
                print(f"  - {file.name}", file=sys.stderr)
            return False
        
        print(f"STATUS: Found RVC output: {rvc_output.name}", file=sys.stderr)
        current_source = rvc_output
        
        # Apply silence cleaning if enabled
        if rvc_config['clean_silence']:
            print(f"STATUS: Removing long pauses", file=sys.stderr)
            temp_cleaned = temp_path / "cleaned_output.wav"
            if remove_long_silence(
                current_source, 
                temp_cleaned, 
                rvc_config['silence_threshold'], 
                rvc_config['silence_duration']
            ):
                current_source = temp_cleaned
                print("STATUS: Pause removal completed", file=sys.stderr)
            else:
                print("WARNING: Pause removal failed, using original RVC output", file=sys.stderr)
        
        # Speed up the audio if requested
        if rvc_config['speed_factor'] != 1.0:
            print(f"STATUS: Adjusting speed by {rvc_config['speed_factor']}x", file=sys.stderr)
            temp_spedup = temp_path / "spedup_output.wav"
            if speed_up_audio(current_source, temp_spedup, rvc_config['speed_factor']):
                current_source = temp_spedup
                print("STATUS: Speed adjustment completed", file=sys.stderr)
            else:
                print("WARNING: Speed adjustment failed, using cleaned audio", file=sys.stderr)
        
        # Copy final result to output location
        shutil.copy2(current_source, output_path)
        print(f"STATUS: RVC processing completed: {output_path}", file=sys.stderr)
    
    return True

def process_combined_audio(combined_file, final_file, config, skip_rvc=False):
    """Process combined audio through RVC or just copy if skipping"""
    
    if skip_rvc:
        print(f"STATUS: Skipping RVC conversion", file=sys.stderr)
        shutil.copy2(combined_file, final_file)
        print(f"STATUS: Final audio: {final_file}", file=sys.stderr)
        return True
    
    # Process through RVC with core configuration only
    return process_audio_through_rvc(combined_file, final_file, config)

def ensure_audio_config(config):
    """Ensure audio config exists in the main config with core supported settings"""
    defaults = get_audio_default_config()
    config_updated = False
    
    for section, section_config in defaults.items():
        if section not in config:
            config[section] = section_config.copy()
            config_updated = True
            print(f"STATUS: Added default {section} config", file=sys.stderr)
        else:
            for key, value in section_config.items():
                if key not in config[section]:
                    config[section][key] = value
                    config_updated = True
                    print(f"STATUS: Added default {section}.{key}: {value}", file=sys.stderr)
    
    return config_updated