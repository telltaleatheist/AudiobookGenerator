#!/usr/bin/env python3
"""
Engine Registry - FIXED: No automatic config modifications
The config file is the single source of truth
"""

import sys
import json
from pathlib import Path

# Available engines will register themselves
_ENGINES = {}

def register_engine(name, processor_func, default_config):
    """Register a TTS engine with its processor function and default config"""
    _ENGINES[name] = {
        'processor': processor_func,
        'default_config': default_config
    }
    print(f"STATUS: Registered TTS engine: {name}", file=sys.stderr)

def get_available_engines():
    """Get list of available engine names"""
    return list(_ENGINES.keys())

def get_engine_processor(engine_name):
    """Get the processor function for an engine"""
    if engine_name not in _ENGINES:
        available = ', '.join(get_available_engines())
        raise ValueError(f"Unknown TTS engine: {engine_name}. Available: {available}")
    
    return _ENGINES[engine_name]['processor']

def ensure_engine_config(config, engine_name):
    """REMOVED: No longer adds missing config sections
    
    The config file is the single source of truth.
    If a section is missing, the engine should handle it gracefully
    or the user should add it to their config file.
    """
    # FIXED: Always return False - never modify config
    print(f"STATUS: Using {engine_name} config AS-IS from config file", file=sys.stderr)
    
    if engine_name not in config:
        print(f"WARNING: No {engine_name} section in config file", file=sys.stderr)
        print(f"INFO: Engine will use its own defaults for missing parameters", file=sys.stderr)
    else:
        section_keys = list(config[engine_name].keys())
        print(f"STATUS: Found {len(section_keys)} {engine_name} parameters in config", file=sys.stderr)
        print(f"STATUS: {engine_name} config keys: {', '.join(section_keys)}", file=sys.stderr)
    
    return False  # Never indicates config was updated

def process_with_engine(engine_name, text_file, output_dir, config, paths):
    """Process text file with the specified engine - NO config modification"""
    processor = get_engine_processor(engine_name)
    
    # REMOVED: No longer calls ensure_engine_config()
    # The config file is used AS-IS
    
    print(f"STATUS: Processing with {engine_name.upper()} engine", file=sys.stderr)
    print(f"STATUS: Config file is single source of truth", file=sys.stderr)
    
    # Call the engine processor with config AS-IS
    return processor(text_file, output_dir, config, paths)

# Import and register engines when this module is loaded
def _register_available_engines():
    """Import and register all available engines"""
    try:
        from bark_engine import register_bark_engine
        register_bark_engine()
    except ImportError as e:
        print(f"WARNING: Bark engine not available: {e}", file=sys.stderr)
    
    try:
        from edge_engine import register_edge_engine
        register_edge_engine()
    except ImportError as e:
        print(f"WARNING: Edge engine not available: {e}", file=sys.stderr)
    
    try:
        from f5_engine import register_f5_engine
        register_f5_engine()
    except ImportError as e:
        print(f"WARNING: F5 engine not available: {e}", file=sys.stderr)
    
    try:
        from xtts_engine import register_xtts_engine
        register_xtts_engine()
    except ImportError as e:
        print(f"WARNING: XTTS engine not available: {e}", file=sys.stderr)

# Auto-register engines on import
_register_available_engines()