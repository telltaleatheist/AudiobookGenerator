#!/usr/bin/env python3
"""
Audio module - Audio processing and combination
"""

from .audio_combiner import combine_audio_files, combine_master_file, get_audio_duration, normalize_audio
from .rvc_processor import process_audio_through_rvc

__all__ = [
    'combine_audio_files',
    'combine_master_file', 
    'get_audio_duration',
    'normalize_audio',
    'process_audio_through_rvc'
]
