#!/usr/bin/env python3
"""
Engine Registry - Simple routing system for TTS engines
Handles engine registration and config defaults
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
    """Ensure engine config exists, add defaults if missing"""
    if engine_name not in _ENGINES:
        raise ValueError(f"Unknown TTS engine: {engine_name}")
    
    engine_info = _ENGINES[engine_name]
    default_config = engine_info['default_config']
    
    # Add missing config sections
    config_updated = False
    for section, defaults in default_config.items():
        if section not in config:
            config[section] = defaults.copy()
            config_updated = True
            print(f"STATUS: Added default {section} config for {engine_name}", file=sys.stderr)
        else:
            # Add missing keys within existing sections
            for key, value in defaults.items():
                if key not in config[section]:
                    config[section][key] = value
                    config_updated = True
    
    return config_updated

def process_with_engine(engine_name, text_file, output_dir, config, paths):
    """Process text file with the specified engine"""
    processor = get_engine_processor(engine_name)
    
    # Ensure config has all required settings
    ensure_engine_config(config, engine_name)
    
    print(f"STATUS: Processing with {engine_name.upper()} engine", file=sys.stderr)
    
    # Call the engine processor
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