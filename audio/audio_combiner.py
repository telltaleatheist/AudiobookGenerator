#!/usr/bin/env python3
"""
Audio Combiner - FFmpeg-based audio combination and file operations
IMPROVED: Better error reporting and Windows path handling
"""

import sys
import subprocess
import tempfile
import os
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
                # Use forward slashes for cross-platform compatibility
                file_path = Path(audio_file).absolute().as_posix()
                f.write(f"file '{file_path}'\n")
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
            print_above_progress("Audio files combined successfully", "success")
            return True
            
        finally:
            # Clean up concat file
            try:
                Path(concat_file).unlink()
            except:
                pass
        
    except subprocess.CalledProcessError as e:
        log_error(f"FFmpeg failed to combine audio files. Exit code: {e.returncode}")
        if e.stderr:
            log_error(f"FFmpeg error: {e.stderr}")
        return False
    except FileNotFoundError:
        log_error("ffmpeg not found. Please install ffmpeg")
        return False

def convert_to_mp3(input_file: str, output_file: str, bitrate: int = 320) -> bool:
    """Convert audio file to high-quality MP3"""
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_file),
            "-codec:a", "libmp3lame",
            "-b:a", f"{bitrate}k",
            "-q:a", "0",  # Highest quality setting for libmp3lame
            str(output_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
        
    except subprocess.CalledProcessError as e:
        log_error(f"MP3 conversion failed. Exit code: {e.returncode}")
        if e.stderr:
            log_error(f"FFmpeg error: {e.stderr}")
        return False
    except FileNotFoundError:
        log_error("ffmpeg not found. Please install ffmpeg")
        return False
    
def combine_master_file(section_file: str, master_file: str) -> bool:
    """Add a section to the master file, creating it if it doesn't exist - WITH BETTER ERROR REPORTING"""
    
    section_path = Path(section_file)
    master_path = Path(master_file)
    
    print_above_progress(f"DEBUG: Combining section {section_path.name} with master", "info")
    
    if not section_path.exists():
        log_error(f"Section file not found: {section_path}")
        return False
    
    # Verify section file is readable
    try:
        file_size = section_path.stat().st_size
        print_above_progress(f"DEBUG: Section file size: {file_size:,} bytes", "info")
        if file_size == 0:
            log_error(f"Section file is empty: {section_path}")
            return False
    except Exception as e:
        log_error(f"Cannot access section file: {e}")
        return False
    
    # If master doesn't exist, just copy the section
    if not master_path.exists():
        try:
            print_above_progress("DEBUG: Creating initial master file", "info")
            import shutil
            shutil.copy2(section_path, master_path)
            print_above_progress("Created master file with first section", "success")
            return True
        except Exception as e:
            log_error(f"Failed to create initial master file: {e}")
            return False
    
    # Master exists, append the new section
    print_above_progress("DEBUG: Master file exists, appending new section", "info")
    
    try:
        # Verify master file is readable
        master_size = master_path.stat().st_size
        print_above_progress(f"DEBUG: Current master file size: {master_size:,} bytes", "info")
        
        # Create temporary combined file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
        
        print_above_progress(f"DEBUG: Using temp file: {temp_path}", "info")
        
        # Create concat file with proper path formatting
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            # Use forward slashes for cross-platform compatibility
            master_posix = master_path.absolute().as_posix()
            section_posix = section_path.absolute().as_posix()
            
            f.write(f"file '{master_posix}'\n")
            f.write(f"file '{section_posix}'\n")
            concat_file = f.name
        
        print_above_progress(f"DEBUG: Created concat file: {concat_file}", "info")
        
        # Debug: Show concat file contents
        try:
            with open(concat_file, 'r', encoding='utf-8') as f:
                concat_contents = f.read()
            print_above_progress(f"DEBUG: Concat file contents:\n{concat_contents}", "info")
        except Exception as e:
            print_above_progress(f"DEBUG: Could not read concat file: {e}", "info")
        
        try:
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file,
                "-c", "copy",
                temp_path
            ]
            
            print_above_progress(f"DEBUG: Running FFmpeg command: {' '.join(cmd)}", "info")
            
            # Run without suppressing stderr so we can see what's wrong
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            print_above_progress("DEBUG: FFmpeg completed successfully", "info")
            
            # Verify temp file was created and has content
            temp_path_obj = Path(temp_path)
            if not temp_path_obj.exists():
                log_error("Temp file was not created by FFmpeg")
                return False
            
            temp_size = temp_path_obj.stat().st_size
            print_above_progress(f"DEBUG: Temp file size: {temp_size:,} bytes", "info")
            
            if temp_size == 0:
                log_error("Temp file is empty after FFmpeg processing")
                return False
            
            # Replace master with combined file
            print_above_progress("DEBUG: Replacing master file with combined result", "info")
            import shutil
            
            # Backup original master in case something goes wrong
            backup_path = master_path.with_suffix('.backup.wav')
            try:
                shutil.copy2(master_path, backup_path)
                print_above_progress(f"DEBUG: Created backup: {backup_path}", "info")
            except Exception as e:
                print_above_progress(f"DEBUG: Could not create backup: {e}", "info")
            
            # Move temp to master
            shutil.move(temp_path, master_path)
            
            # Verify final result
            final_size = master_path.stat().st_size
            print_above_progress(f"DEBUG: Final master file size: {final_size:,} bytes", "info")
            
            # Clean up backup if everything worked
            try:
                if backup_path.exists():
                    backup_path.unlink()
            except:
                pass
            
            print_above_progress("Added section to master file", "success")
            return True
            
        finally:
            # Clean up temporary files
            try:
                Path(concat_file).unlink()
                print_above_progress("DEBUG: Cleaned up concat file", "info")
            except Exception as e:
                print_above_progress(f"DEBUG: Could not clean up concat file: {e}", "info")
            
            try:
                temp_path_obj = Path(temp_path)
                if temp_path_obj.exists():
                    temp_path_obj.unlink()
                    print_above_progress("DEBUG: Cleaned up temp file", "info")
            except Exception as e:
                print_above_progress(f"DEBUG: Could not clean up temp file: {e}", "info")
        
    except subprocess.CalledProcessError as e:
        log_error(f"FFmpeg failed to combine with master. Exit code: {e.returncode}")
        if e.stdout:
            log_error(f"FFmpeg stdout: {e.stdout}")
        if e.stderr:
            log_error(f"FFmpeg stderr: {e.stderr}")
        return False
    except Exception as e:
        log_error(f"Master file combination error: {e}")
        import traceback
        log_error(f"Traceback: {traceback.format_exc()}")
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
        log_error(f"Could not get audio duration for {audio_file}: {e}")
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
        log_error(f"Audio normalization failed: {e}")
        if e.stderr:
            log_error(f"FFmpeg error: {e.stderr}")
        return False
    except FileNotFoundError:
        log_error("ffmpeg not found")
        return False
