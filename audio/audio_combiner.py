#!/usr/bin/env python3
"""
Audio Combiner - FFmpeg-based audio combination and file operations
"""

import sys
import subprocess
import tempfile
from pathlib import Path
from typing import List

def combine_audio_files(audio_files: List[str], output_path: str, silence_gap: float = 0.3) -> bool:
    """Combine multiple audio files with silence gaps"""
    
    if not audio_files:
        print(f"❌ No audio files to combine", file=sys.stderr)
        return False
    
    if len(audio_files) == 1:
        # Single file, just copy it
        import shutil
        shutil.copy2(audio_files[0], output_path)
        return True
    
    print(f"STATUS: Combining {len(audio_files)} audio files", file=sys.stderr)
    
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
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"STATUS: Audio files combined successfully", file=sys.stderr)
            return True
            
        finally:
            # Clean up concat file
            try:
                Path(concat_file).unlink()
            except:
                pass
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to combine audio files: {e}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print(f"❌ ffmpeg not found. Please install ffmpeg", file=sys.stderr)
        return False

def combine_master_file(section_file: str, master_file: str) -> bool:
    """Add a section to the master file, creating it if it doesn't exist"""
    
    section_path = Path(section_file)
    master_path = Path(master_file)
    
    if not section_path.exists():
        print(f"❌ Section file not found: {section_path}", file=sys.stderr)
        return False
    
    # If master doesn't exist, just copy the section
    if not master_path.exists():
        import shutil
        shutil.copy2(section_path, master_path)
        print(f"STATUS: Created master file with first section", file=sys.stderr)
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
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Replace master with combined file
            import shutil
            shutil.move(temp_path, master_path)
            
            print(f"STATUS: Added section to master file", file=sys.stderr)
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
        print(f"❌ Failed to combine with master: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"❌ Master file combination error: {e}", file=sys.stderr)
        return False

def get_audio_duration(audio_file: str) -> float:
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
        print(f"❌ Could not get audio duration for {audio_file}: {e}", file=sys.stderr)
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
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Audio normalization failed: {e}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print(f"❌ ffmpeg not found", file=sys.stderr)
        return False