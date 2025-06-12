#!/usr/bin/env python3
"""
Engines Package - TTS Engine Registry
Automatically imports and registers all available engines
"""

# Import base engine functionality
from core.progress_display_manager import log_info
from .base_engine import (
    get_available_engines,
    get_engine_processor, 
    register_engine,
    process_with_engine
)

# Import and register all engines
def register_all_engines():
    """Register all available TTS engines"""
    
    # Always register XTTS (working engine)
    try:
        from .xtts_engine import register_xtts_engine
        register_xtts_engine()
    except ImportError as e:
        log_info("Warning: Could not register XTTS engine: {e}")
    
    # Register OpenAI engine
    try:
        from .openai_engine import register_openai_engine
        register_openai_engine()
    except ImportError as e:
        log_info("Warning: Could not register OpenAI engine: {e}")
    
    # Register Edge engine
    try:
        from .edge_engine import register_edge_engine
        register_edge_engine()
    except ImportError as e:
        log_info("Warning: Could not register Edge engine: {e}")
    
    # Register F5 engine
    try:
        from .f5_engine import register_f5_engine
        register_f5_engine()
    except ImportError as e:
        log_info("Warning: Could not register F5 engine: {e}")
    
    # Register Bark engine (when available)
    try:
        from .bark_engine import register_bark_engine
        register_bark_engine()
    except ImportError as e:
        log_info("Warning: Could not register Bark engine: {e}")

# Register all engines when package is imported
register_all_engines()

# Export main functions
__all__ = [
    'get_available_engines',
    'get_engine_processor',
    'register_engine',
    'process_with_engine'
]
