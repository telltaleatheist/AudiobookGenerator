#!/usr/bin/env python3
"""
RVC Processor - CLEANED: Simplified audio processing without verbose commands
FIXED: Added proper imports and string formatting
"""

import sys
import subprocess
import os
import tempfile
import shutil
import warnings
from pathlib import Path
from core.progress_display_manager import log_error, log_info, print_above_progress
from managers.config_manager import ConfigManager, ConfigError

# Suppress warnings that interfere with progress bars
warnings.filterwarnings("ignore")

def process_audio_through_rvc(input_file: str, output_file: str, config: dict) -> bool:
    """Process audio through RVC conversion - CLEANED: Less verbose output"""
    
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    try:
        # Get RVC voice from metadata
        rvc_voice = config['metadata']['rvc_voice']
        # CLEANED: Remove verbose voice processing message
        
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
        
        # CLEANED: Remove verbose model message
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Run RVC conversion
        success = _run_rvc_conversion(input_path, output_path, rvc_config)
        
        if success:
            # CLEANED: Remove verbose completion message
            return True
        else:
            print_above_progress("RVC processing failed", "error")
            return False
            
    except ConfigError as e:
        log_info(f"RVC Configuration Error: {e}", "error")
        return False
    except Exception as e:
        log_error("RVC processing error")
        return False

def _run_rvc_conversion(input_path: Path, output_path: Path, rvc_config: dict) -> bool:
    """Run the actual RVC conversion - CLEANED: No command echo"""
    
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
        
        # CLEANED: Remove command echo - too verbose
        
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
                log_error("Could not find RVC output file")
                return False
            
            # Apply post-processing
            processed_file = _apply_rvc_post_processing(rvc_output, rvc_config)
            
            # Copy final result to output location
            shutil.copy2(processed_file, output_path)
            
            return True
            
        except subprocess.CalledProcessError as e:
            log_info(f"RVC conversion failed: {e}", "error")
            if e.stderr:
                log_info("RVC Error")
            return False
        except FileNotFoundError:
            print_above_progress("RVC command 'urvc' not found. Make sure RVC is installed and in PATH", "error")
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
        log_error("Speed adjustment failed")
        return False
    except FileNotFoundError:
        log_error("ffmpeg not found")
        return False
