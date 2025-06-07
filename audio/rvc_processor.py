#!/usr/bin/env python3
"""
RVC Processor - Simplified audio processing without splitting logic
Processes complete sections through RVC conversion
"""

import sys
import subprocess
import os
import tempfile
import shutil
from pathlib import Path
from managers.config_manager import ConfigManager, ConfigError

def process_audio_through_rvc(input_file: str, output_file: str, config: dict) -> bool:
    """Process audio through RVC conversion - simplified for section-based pipeline"""
    
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    try:
        # Get RVC voice from metadata
        rvc_voice = config['metadata']['rvc_voice']
        print(f"STATUS: RVC processing with voice: {rvc_voice}", file=sys.stderr)
        
        # Get voice-specific config
        rvc_voice_key = f'rvc_{rvc_voice}'
        if rvc_voice_key not in config:
            raise ConfigError(f"RVC voice config '{rvc_voice_key}' not found")
        
        # Combine global + voice-specific settings
        rvc_global = config.get('rvc_global', {})
        rvc_voice_config = config[rvc_voice_key]
        
        # Voice-specific settings override global settings
        rvc_config = {**rvc_global, **rvc_voice_config}
        
        # Validate required RVC settings
        required_settings = ['model', 'f0_method']
        for setting in required_settings:
            if setting not in rvc_config:
                raise ConfigError(f"Missing required RVC setting: {setting}")
        
        print(f"STATUS: RVC model: {rvc_config['model']}", file=sys.stderr)
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Run RVC conversion
        success = _run_rvc_conversion(input_path, output_path, rvc_config)
        
        if success:
            print(f"STATUS: RVC processing completed: {output_path.name}", file=sys.stderr)
            return True
        else:
            print(f"❌ RVC processing failed", file=sys.stderr)
            return False
            
    except ConfigError as e:
        print(f"❌ RVC Configuration Error: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"❌ RVC processing error: {e}", file=sys.stderr)
        return False

def _run_rvc_conversion(input_path: Path, output_path: Path, rvc_config: dict) -> bool:
    """Run the actual RVC conversion"""
    
    # Create temporary directory for RVC processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Build RVC command
        cmd = [
            "urvc", "generate", "convert-voice",
            str(input_path),
            str(temp_path),
            rvc_config['model']
        ]
        
        # Add core RVC parameters
        cmd.extend([
            "--f0-method", rvc_config.get('f0_method', 'crepe'),
            "--hop-length", str(rvc_config.get('hop_length', 64))
        ])
        
        # Add optional parameters if present
        if 'n_semitones' in rvc_config:
            cmd.extend(["--n-semitones", str(rvc_config['n_semitones'])])
        
        if 'index_rate' in rvc_config:
            cmd.extend(["--index-rate", str(rvc_config['index_rate'])])
        
        if 'protect_rate' in rvc_config:
            cmd.extend(["--protect-rate", str(rvc_config['protect_rate'])])
        
        if 'rms_mix_rate' in rvc_config:
            cmd.extend(["--rms-mix-rate", str(rvc_config['rms_mix_rate'])])
        
        # Add boolean flags
        if rvc_config.get('split_voice', False):
            cmd.append("--split-voice")
        
        if rvc_config.get('clean_voice', False):
            cmd.append("--clean-voice")
            if 'clean_strength' in rvc_config:
                cmd.extend(["--clean-strength", str(rvc_config['clean_strength'])])
        
        if rvc_config.get('autotune_voice', False):
            cmd.append("--autotune-voice")
            if 'autotune_strength' in rvc_config:
                cmd.extend(["--autotune-strength", str(rvc_config['autotune_strength'])])
        
        print(f"STATUS: Running RVC command: {' '.join(cmd)}", file=sys.stderr)
        
        try:
            # Set up environment
            env = os.environ.copy()
            env['PYTHONHASHSEED'] = '42'
            
            # Run RVC
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True,
                env=env
            )
            
            # Find RVC output file
            rvc_output = _find_rvc_output_file(temp_path, input_path.stem)
            
            if not rvc_output:
                print(f"❌ Could not find RVC output file in {temp_path}", file=sys.stderr)
                return False
            
            # Apply post-processing
            processed_file = _apply_rvc_post_processing(rvc_output, rvc_config)
            
            # Copy final result to output location
            shutil.copy2(processed_file, output_path)
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ RVC conversion failed: {e}", file=sys.stderr)
            if e.stderr:
                print(f"RVC Error: {e.stderr}", file=sys.stderr)
            return False
        except FileNotFoundError:
            print(f"❌ RVC command 'urvc' not found. Make sure RVC is installed and in PATH", file=sys.stderr)
            return False

def _find_rvc_output_file(output_dir: Path, base_name: str) -> Path:
    """Find the RVC output file"""
    
    # Common patterns RVC might use
    patterns = [
        "*.wav",
        "converted_*.wav",
        "*_converted.wav",
        "*Voice_Converted*.wav",
        f"{base_name}*.wav",
        f"*{base_name}*.wav"
    ]
    
    for pattern in patterns:
        matches = list(output_dir.glob(pattern))
        if matches:
            # Return the most recent file if multiple matches
            return max(matches, key=lambda p: p.stat().st_mtime)
    
    return None

def _apply_rvc_post_processing(input_file: Path, rvc_config: dict) -> Path:
    """Apply post-processing effects to RVC output"""
    
    # For now, just return the input file
    # Could add normalization, silence removal, etc. here
    
    return input_file

def _apply_speed_adjustment(input_file: Path, output_file: Path, speed_factor: float) -> bool:
    """Apply speed adjustment using ffmpeg"""
    if speed_factor == 1.0:
        shutil.copy2(input_file, output_file)
        return True
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_file),
        "-af", f"atempo={speed_factor}",
        str(output_file)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Speed adjustment failed: {e}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print(f"❌ ffmpeg not found", file=sys.stderr)
        return False