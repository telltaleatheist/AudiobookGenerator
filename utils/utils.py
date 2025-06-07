#!/usr/bin/env python3
"""
Utils - Utility functions and helpers
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from core.progress_display_manager import log_error

def format_duration(seconds: float) -> str:
    """Format duration in seconds to HH:MM:SS.MS format"""
    ms = int((seconds - int(seconds)) * 100)
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f"{h:02}:{m:02}:{s:02}.{ms:02}"

def time_function(func, *args, **kwargs):
    """Time a function execution and return result with duration"""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    duration = end_time - start_time
    return result, duration

def safe_file_operation(operation, file_path: Path, *args, **kwargs):
    """Safely perform file operations with error handling"""
    try:
        return operation(file_path, *args, **kwargs)
    except Exception as e:
        log_error(f"File operation failed on {file_path}")
        return None

def ensure_directory(path: Path) -> bool:
    """Ensure directory exists, create if needed"""
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        log_error(f"Could not create directory {path}")
        return False

def load_json_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file with error handling"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        log_error(f"Could not load JSON file {file_path}")
        return None

def save_json_file(data: Dict[str, Any], file_path: Path) -> bool:
    """Save data to JSON file with error handling"""
    try:
        ensure_directory(file_path.parent)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        log_error(f"Could not save JSON file {file_path}")
        return False

def get_file_size_mb(file_path: Path) -> float:
    """Get file size in megabytes"""
    try:
        return file_path.stat().st_size / (1024 * 1024)
    except Exception:
        return 0.0

def clean_filename(filename: str) -> str:
    """Clean filename for safe filesystem usage"""
    import re
    # Replace problematic characters
    cleaned = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove multiple underscores
    cleaned = re.sub(r'_+', '_', cleaned)
    # Remove leading/trailing underscores
    cleaned = cleaned.strip('_')
    return cleaned

def truncate_string(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """Truncate string to max length with suffix"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def log_to_file(message: str, log_file: Path, timestamp: bool = True):
    """Log message to file with optional timestamp"""
    try:
        ensure_directory(log_file.parent)
        with open(log_file, 'a', encoding='utf-8') as f:
            if timestamp:
                ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"[{ts}] {message}\n")
            else:
                f.write(f"{message}\n")
    except Exception as e:
        log_error(f"Could not write to log file {log_file}")

def estimate_words_per_minute(text: str, wpm: int = 150) -> float:
    """Estimate reading time in minutes based on word count"""
    word_count = len(text.split())
    return word_count / wpm

def validate_audio_file(file_path: Path) -> bool:
    """Validate that a file is a supported audio format"""
    if not file_path.exists():
        return False
    
    supported_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
    return file_path.suffix.lower() in supported_extensions

def get_text_preview(text: str, max_chars: int = 100) -> str:
    """Get a preview of text content"""
    if len(text) <= max_chars:
        return text
    
    # Try to break at word boundary
    truncated = text[:max_chars]
    last_space = truncated.rfind(' ')
    
    if last_space > max_chars * 0.8:  # If we found a space reasonably close to the end
        truncated = truncated[:last_space]
    
    return truncated + "..."