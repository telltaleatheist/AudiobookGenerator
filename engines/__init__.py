#!/usr/bin/env python3
"""
Engines module - TTS engine registry and exports
"""

# Import registry functions from base_engine
from .base_engine import (
    register_engine,
    get_available_engines, 
    get_engine_processor,
    process_with_engine
)

# Import and register available engines
try:
    from .xtts_engine import register_xtts_engine
    register_xtts_engine()
    print("STATUS: XTTS engine registered")
except ImportError as e:
    print(f"WARNING: Could not register XTTS engine: {e}")

# TODO: Register other engines when they're updated
# from .bark_engine import register_bark_engine
# from .edge_engine import register_edge_engine  
# from .f5_engine import register_f5_engine
# from .openai_engine import register_openai_engine

# register_bark_engine()
# register_edge_engine()
# register_f5_engine() 
# register_openai_engine()

# Export main functions
__all__ = [
    'get_engine_processor',
    'get_available_engines', 
    'process_with_engine',
    'register_engine'
]