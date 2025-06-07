#!/usr/bin/env python3
"""
Audio Combiner - FFmpeg-based audio combination and file operations
FIXED: Added proper imports and string formatting
"""

import sys
import subprocess
import tempfile
from pathlib import Path
from typing import List

from core.progress_display_manager import log_error, log_info, print_above_progress

def combine_audio_files(audio_files: List[str], output_path: str, silence_gap: float = 0.3) -> bool:
    """Combine multiple audio files with silence gaps"""
    
    if not audio_files:
        log_error("No audio files to combine")
        return False
    
    if len(audio_files) == 1:
        # Single file, just copy it
        import shutil
        shutil.copy2(audio_files[0], output_path)
        return True
    
    log_info(f"Combining {len(audio_files)} audio files")
        
    try:
        # Create temporary concat file for ffmpeg
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for audio_file in audio_files:
                f.write(f"file '{Path(audio_file).absolute()}'\n")
            concat_file = f.name
        
        try:
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file,
                "-c", "copy",
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print_above_progress("Audio files combined successfully", "success")
            return True
            
        finally:
            # Clean up concat file
            try:
                Path(concat_file).unlink()
            except:
                pass
        
    except subprocess.CalledProcessError as e:
        log_error("Failed to combine audio files")
        return False
    except FileNotFoundError:
        log_error("ffmpeg not found. Please install ffmpeg")
        return False

def combine_master_file(section_file: str, master_file: str) -> bool:
    """Add a section to the master file, creating it if it doesn't exist"""
    
    section_path = Path(section_file)
    master_path = Path(master_file)
    
    if not section_path.exists():
        log_error("Section file not found")
        return False
    
    # If master doesn't exist, just copy the section
    if not master_path.exists():
        import shutil
        shutil.copy2(section_path, master_path)
        print_above_progress("Created master file with first section", "success")
        return True
    
    # Master exists, append the new section
    try:
        # Create temporary combined file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Use ffmpeg to concatenate
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(f"file '{master_path.absolute()}'\n")
            f.write(f"file '{section_path.absolute()}'\n")
            concat_file = f.name
        
        try:
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file,
                "-c", "copy",
                temp_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Replace master with combined file
            import shutil
            shutil.move(temp_path, master_path)
            
            print_above_progress("Added section to master file", "success")
            return True
            
        finally:
            # Clean up
            try:
                Path(concat_file).unlink()
            except:
                pass
            try:
                Path(temp_path).unlink()
            except:
                pass
        
    except subprocess.CalledProcessError as e:
        log_info("Failed to combine with master")
        return False
    except Exception as e:
        log_info("Master file combination error")
        return False

def get_audio_duration(audio_file: str) -> float:
    """Get audio duration in seconds using ffprobe"""
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(audio_file)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        duration = float(result.stdout.strip())
        return duration
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
        log_error(f"Could not get audio duration for {audio_file}")
        return 0.0

def normalize_audio(input_file: str, output_file: str) -> bool:
    """Normalize audio levels using ffmpeg"""
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_file),
            "-af", "loudnorm",
            str(output_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
        
    except subprocess.CalledProcessError as e:
        log_error("Audio normalization failed")
        return False
    except FileNotFoundError:
        log_error("ffmpeg not found")
        return False