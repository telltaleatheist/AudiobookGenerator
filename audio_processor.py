#!/usr/bin/env python3
"""
Audio Processor - Handles RVC conversion and final audio cleanup
ENHANCED: Now includes automatic splitting for large files to prevent RVC crashes
"""

import sys
import subprocess
import tempfile
import shutil
import os
from pathlib import Path
import json

def get_audio_duration(audio_file):
    """Get audio duration in seconds using ffprobe"""
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(audio_file)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        return duration
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
        print(f"ERROR: Could not get audio duration for {audio_file}: {e}", file=sys.stderr)
        return None

def find_silence_points(audio_file, num_segments=2, min_silence_duration=0.5, silence_threshold=-40):
    """Find optimal silence points to split audio into equal segments"""
    try:
        duration = get_audio_duration(audio_file)
        if duration is None:
            return []
        
        # Calculate ideal split points
        segment_duration = duration / num_segments
        ideal_split_points = [segment_duration * i for i in range(1, num_segments)]
        
        print(f"STATUS: Looking for silence points near {ideal_split_points} seconds", file=sys.stderr)
        
        # Use ffmpeg silencedetect to find all silence periods
        cmd = [
            "ffmpeg", "-i", str(audio_file), "-af", 
            f"silencedetect=noise={silence_threshold}dB:duration={min_silence_duration}",
            "-f", "null", "-"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse silence periods from ffmpeg output
        silence_periods = []
        lines = result.stderr.split('\n')
        
        silence_start = None
        for line in lines:
            if "silence_start:" in line:
                try:
                    silence_start = float(line.split("silence_start:")[1].split()[0])
                except (IndexError, ValueError):
                    continue
            elif "silence_end:" in line and silence_start is not None:
                try:
                    silence_end = float(line.split("silence_end:")[1].split()[0])
                    silence_duration = silence_end - silence_start
                    if silence_duration >= min_silence_duration:
                        silence_periods.append((silence_start, silence_end))
                    silence_start = None
                except (IndexError, ValueError):
                    continue
        
        if not silence_periods:
            print(f"WARNING: No suitable silence periods found, using approximate splits", file=sys.stderr)
            return ideal_split_points
        
        # Find best silence period for each ideal split point
        actual_split_points = []
        search_window = 30.0  # Search within 30 seconds of ideal point
        
        for ideal_point in ideal_split_points:
            best_silence = None
            best_distance = float('inf')
            
            for silence_start, silence_end in silence_periods:
                silence_mid = (silence_start + silence_end) / 2
                distance = abs(silence_mid - ideal_point)
                
                if distance <= search_window and distance < best_distance:
                    best_distance = distance
                    best_silence = silence_mid
            
            if best_silence is not None:
                actual_split_points.append(best_silence)
                print(f"STATUS: Found silence split at {best_silence:.1f}s (target: {ideal_point:.1f}s)", file=sys.stderr)
            else:
                # Fallback to ideal point if no silence found
                actual_split_points.append(ideal_point)
                print(f"WARNING: No silence near {ideal_point:.1f}s, using approximate split", file=sys.stderr)
        
        return actual_split_points
        
    except Exception as e:
        print(f"ERROR: Failed to find silence points: {e}", file=sys.stderr)
        # Fallback to simple time-based splits
        duration = get_audio_duration(audio_file)
        if duration:
            segment_duration = duration / num_segments
            return [segment_duration * i for i in range(1, num_segments)]
        return []

def split_audio_at_points(audio_file, split_points, output_dir):
    """Split audio file at specified points"""
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        duration = get_audio_duration(audio_file)
        if duration is None:
            raise ValueError("Could not determine audio duration")
        
        # Create segments based on split points
        segments = []
        start_times = [0.0] + split_points
        end_times = split_points + [duration]
        
        for i, (start_time, end_time) in enumerate(zip(start_times, end_times)):
            segment_num = i + 1
            output_file = output_dir / f"segment_{segment_num:02d}.wav"
            
            segment_duration = end_time - start_time
            print(f"STATUS: Creating segment {segment_num}: {start_time:.1f}s - {end_time:.1f}s ({segment_duration:.1f}s)", file=sys.stderr)
            
            cmd = [
                "ffmpeg", "-y", "-i", str(audio_file),
                "-ss", str(start_time),
                "-t", str(segment_duration),
                "-c", "copy",  # Copy without re-encoding when possible
                str(output_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            segments.append(output_file)
            
            print(f"STATUS: Segment {segment_num} created: {output_file.name}", file=sys.stderr)
        
        return segments
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to split audio: {e}", file=sys.stderr)
        print(f"STDERR: {e.stderr}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"ERROR: Audio splitting failed: {e}", file=sys.stderr)
        return []

def combine_audio_segments(segment_files, output_file):
    """Combine audio segments back into single file"""
    try:
        if not segment_files:
            print(f"ERROR: No segments to combine", file=sys.stderr)
            return False
        
        if len(segment_files) == 1:
            # Single segment, just copy
            shutil.copy2(segment_files[0], output_file)
            print(f"STATUS: Single segment copied to {output_file}", file=sys.stderr)
            return True
        
        print(f"STATUS: Combining {len(segment_files)} segments", file=sys.stderr)
        
        # Create temporary concat file for ffmpeg
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for segment_file in segment_files:
                f.write(f"file '{Path(segment_file).absolute()}'\n")
            concat_file = f.name
        
        try:
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file,
                "-c", "copy",
                str(output_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"STATUS: Segments combined successfully: {output_file}", file=sys.stderr)
            return True
            
        finally:
            # Clean up concat file
            try:
                Path(concat_file).unlink()
            except:
                pass
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to combine segments: {e}", file=sys.stderr)
        print(f"STDERR: {e.stderr}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"ERROR: Segment combination failed: {e}", file=sys.stderr)
        return False

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
            # NEW: Large file splitting configuration
            'max_duration_hours': 1.0,  # Split files longer than 1 hour
            'split_min_silence_duration': 0.5,  # Minimum silence duration to split on
            'split_silence_threshold': -40,  # dB threshold for silence detection
            'split_search_window': 30.0,  # Seconds to search around ideal split point
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
    """Process an audio file through RVC conversion with ENHANCED large file handling"""
    
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    # GET RVC VOICE FROM METADATA (NEW SYSTEM)
    rvc_voice = config.get('metadata', {}).get('rvc_voice', 'my_voice')
    print(f"STATUS: Using RVC voice profile: {rvc_voice}", file=sys.stderr)
    
    # GET VOICE-SPECIFIC CONFIG
    rvc_voice_key = f'rvc_{rvc_voice}'
    if rvc_voice_key not in config:
        print(f"ERROR: RVC voice config '{rvc_voice_key}' not found!", file=sys.stderr)
        print(f"Available RVC configs: {[k for k in config.keys() if k.startswith('rvc_')]}", file=sys.stderr)
        return False
    
    # COMBINE GLOBAL + VOICE-SPECIFIC SETTINGS
    rvc_global = config.get('rvc_global', {})
    rvc_voice_config = config[rvc_voice_key]
    
    # Voice-specific settings override global settings
    rvc_config = {**rvc_global, **rvc_voice_config}
    
    print(f"STATUS: RVC model: {rvc_config.get('model', 'unknown')}", file=sys.stderr)
    print(f"STATUS: RVC speed factor: {rvc_config.get('speed_factor', 1.0)}", file=sys.stderr)
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"STATUS: Processing {input_path.name} through RVC", file=sys.stderr)
    
    # NEW: Check if file is too large and needs splitting
    duration = get_audio_duration(input_path)
    max_duration_hours = rvc_config.get('max_duration_hours', 1.0)
    max_duration_seconds = max_duration_hours * 3600
    
    if duration is None:
        print(f"WARNING: Could not determine audio duration, proceeding without splitting", file=sys.stderr)
        split_required = False
    elif duration > max_duration_seconds:
        print(f"STATUS: Audio duration {duration/3600:.2f}h exceeds limit of {max_duration_hours}h", file=sys.stderr)
        print(f"STATUS: Will split into segments for RVC processing", file=sys.stderr)
        split_required = True
    else:
        print(f"STATUS: Audio duration {duration/60:.1f}m is within limits, processing normally", file=sys.stderr)
        split_required = False
    
    # Log core RVC settings being used
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
        
        if split_required:
            # ENHANCED: Process large file in segments
            print(f"STATUS: Splitting large audio file for RVC processing", file=sys.stderr)
            
            # Calculate number of segments needed
            num_segments = int((duration / max_duration_seconds) + 0.5) + 1
            print(f"STATUS: Splitting into {num_segments} segments", file=sys.stderr)
            
            # Create splits directory
            splits_dir = temp_path / "splits"
            splits_dir.mkdir()
            
            # Find optimal split points based on silence
            split_points = find_silence_points(
                input_path, 
                num_segments=num_segments,
                min_silence_duration=rvc_config.get('split_min_silence_duration', 0.5),
                silence_threshold=rvc_config.get('split_silence_threshold', -40)
            )
            
            # Split the audio
            segments = split_audio_at_points(input_path, split_points, splits_dir)
            
            if not segments:
                print(f"ERROR: Failed to split audio, falling back to single-file processing", file=sys.stderr)
                split_required = False
            else:
                print(f"STATUS: Created {len(segments)} audio segments", file=sys.stderr)
                
                # Process each segment through RVC
                processed_segments = []
                
                for i, segment_file in enumerate(segments):
                    segment_num = i + 1
                    print(f"STATUS: Processing segment {segment_num}/{len(segments)} through RVC", file=sys.stderr)
                    
                    # Create output directory for this segment
                    segment_output_dir = temp_path / f"rvc_segment_{segment_num:02d}"
                    segment_output_dir.mkdir()
                    
                    # Run RVC on this segment
                    success, stdout = run_rvc_conversion(
                        segment_file, 
                        segment_output_dir, 
                        rvc_config['model'], 
                        rvc_config
                    )
                    
                    if not success:
                        print(f"ERROR: RVC processing failed for segment {segment_num}", file=sys.stderr)
                        return False
                    
                    # Find RVC output for this segment
                    rvc_segment_output = find_rvc_output_file(segment_output_dir, segment_file.stem)
                    
                    if not rvc_segment_output:
                        print(f"ERROR: Could not find RVC output for segment {segment_num}", file=sys.stderr)
                        return False
                    
                    processed_segments.append(rvc_segment_output)
                    print(f"STATUS: Segment {segment_num} processed: {rvc_segment_output.name}", file=sys.stderr)
                
                # Combine processed segments
                print(f"STATUS: Combining {len(processed_segments)} processed RVC segments", file=sys.stderr)
                combined_rvc_file = temp_path / "combined_rvc.wav"
                
                if not combine_audio_segments(processed_segments, combined_rvc_file):
                    print(f"ERROR: Failed to combine RVC processed segments", file=sys.stderr)
                    return False
                
                print(f"STATUS: RVC segments combined successfully", file=sys.stderr)
                current_source = combined_rvc_file
        
        if not split_required:
            # ORIGINAL: Process normally for smaller files
            print(f"STATUS: Processing audio normally (no splitting required)", file=sys.stderr)
            
            # Run RVC conversion with combined config
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
        if rvc_config.get('clean_silence', False):
            print(f"STATUS: Removing long pauses", file=sys.stderr)
            temp_cleaned = temp_path / "cleaned_output.wav"
            if remove_long_silence(
                current_source, 
                temp_cleaned, 
                rvc_config.get('silence_threshold', -40), 
                rvc_config.get('silence_duration', 0.6)
            ):
                current_source = temp_cleaned
                print("STATUS: Pause removal completed", file=sys.stderr)
            else:
                print("WARNING: Pause removal failed, using original RVC output", file=sys.stderr)
        
        # Speed up the audio if requested
        speed_factor = rvc_config.get('speed_factor', 1.0)
        if speed_factor != 1.0:
            print(f"STATUS: Adjusting speed by {speed_factor}x", file=sys.stderr)
            temp_spedup = temp_path / "spedup_output.wav"
            if speed_up_audio(current_source, temp_spedup, speed_factor):
                current_source = temp_spedup
                print("STATUS: Speed adjustment completed", file=sys.stderr)
            else:
                print("WARNING: Speed adjustment failed, using cleaned audio", file=sys.stderr)
        else:
            print("STATUS: No speed adjustment needed (speed_factor = 1.0)", file=sys.stderr)
        
        # Copy final result to output location
        shutil.copy2(current_source, output_path)
        print(f"STATUS: RVC processing completed: {output_path}", file=sys.stderr)
        
        if split_required:
            final_duration = get_audio_duration(output_path)
            if final_duration:
                print(f"STATUS: Final audio duration: {final_duration/60:.1f}m", file=sys.stderr)
    
    return True

def process_combined_audio(combined_file, final_file, config, skip_rvc=False):
    """Process combined audio through RVC or just copy if skipping"""
    
    if skip_rvc:
        print(f"STATUS: Skipping RVC conversion", file=sys.stderr)
        shutil.copy2(combined_file, final_file)
        print(f"STATUS: Final audio: {final_file}", file=sys.stderr)
        return True
    
    # Process through RVC with enhanced large file handling
    return process_audio_through_rvc(combined_file, final_file, config)

def ensure_audio_config(config):
    """Ensure audio config exists - UPDATED to not add old RVC structure"""
    config_updated = False
    
    # Only add audio section if missing (we don't add RVC anymore)
    if 'audio' not in config:
        config['audio'] = {'silence_gap': 0.3}
        config_updated = True
        print(f"STATUS: Added default audio config", file=sys.stderr)
    else:
        # Add missing audio settings
        audio_defaults = {'silence_gap': 0.3}
        for key, value in audio_defaults.items():
            if key not in config['audio']:
                config['audio'][key] = value
                config_updated = True