#!/usr/bin/env python3
"""
Progress Display Manager - Simplified text-only logging without progress bars
"""

import sys
import time
import threading
from typing import Optional, Dict, Any

class ProgressDisplay:
    """Simplified progress display with text-only updates"""
    
    def __init__(self):
        self.lock = threading.Lock()
        
    def update_section_progress(self, completed: int, total: int, eta: str = 'calculating...'):
        """Update section progress - simplified text output"""
        with self.lock:
            section_percent = (completed / max(total, 1)) * 100
            print(f"📊 Overall Progress: {completed}/{total} sections ({section_percent:.0f}%) - ETA: {eta}")
    
    def update_tts_progress(self, completed: int, total: int, eta: str = 'calculating...', active: bool = True):
        """Update TTS progress - simplified text output"""
        if not active or total == 0:
            return
            
        with self.lock:
            tts_percent = (completed / total) * 100
            print(f"  🎤 TTS Generation: {completed}/{total} chunks ({tts_percent:.0f}%) - ETA: {eta}")
    
    def hide_tts_progress(self):
        """Hide TTS progress bar - now just a no-op"""
        pass
    
    def print_above_progress(self, message: str, message_type: str = "info"):
        """Print message - no longer needs to handle progress bars"""
        with self.lock:
            # Print the message with proper formatting
            if message_type == "warning":
                print(f"⚠️ {message}")
            elif message_type == "error":
                print(f"❌ {message}")
            elif message_type == "success":
                print(f"✅ {message}")
            elif message_type == "info":
                print(f"ℹ️ {message}")
            else:
                print(message)
    
    def final_cleanup(self):
        """Clean up progress display - now just a no-op"""
        pass


# Global progress display instance
_progress_display = None

def get_progress_display() -> ProgressDisplay:
    """Get global progress display instance"""
    global _progress_display
    if _progress_display is None:
        _progress_display = ProgressDisplay()
    return _progress_display

def print_above_progress(message: str, message_type: str = "info"):
    """Print message - now just direct output"""
    get_progress_display().print_above_progress(message, message_type)

def log_info(message: str, message_type: str = "info"):
    """Log info message"""
    get_progress_display().print_above_progress(message, message_type)

def log_success(message: str):
    """Log success message"""
    get_progress_display().print_above_progress(message, "success")

def log_warning(message: str):
    """Log warning message"""
    get_progress_display().print_above_progress(message, "warning")

def log_error(message: str):
    """Log error message"""
    get_progress_display().print_above_progress(message, "error")

def log_status(message: str):
    """Log status message"""
    get_progress_display().print_above_progress(message, "info")