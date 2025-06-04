#!/usr/bin/env python3
"""
Engine Registry with Dynamic Parameter Loading
- Registers TTS engines
- Provides dynamic configuration loading utilities
- Config file is the single source of truth
"""

import sys
import json
import inspect
from pathlib import Path

# Available engines will register themselves
_ENGINES = {}

def register_engine(name, processor_func, default_config=None):
    """Register a TTS engine with its processor function"""
    _ENGINES[name] = {
        'processor': processor_func,
        'default_config': default_config or {}  # Optional, not used for config generation
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

# ============================================================================
# DYNAMIC PARAMETER LOADING UTILITIES
# ============================================================================

def extract_engine_config(config, engine_name, verbose=False):
    """
    Extract and return all parameters for a specific engine from config
    No need to predefine what parameters exist - just uses whatever is in the config
    """
    engine_config = config.get(engine_name, {})
    
    if verbose and engine_config:
        print(f"STATUS: Loaded {len(engine_config)} {engine_name} parameters", file=sys.stderr)
        non_null_params = [(k, v) for k, v in engine_config.items() if v is not None]
        if non_null_params:
            print(f"STATUS: Non-null {engine_name} parameters:", file=sys.stderr)
            for key, value in non_null_params:
                print(f"  {key}: {value}", file=sys.stderr)
    elif not engine_config:
        print(f"WARNING: No {engine_name} section found in config", file=sys.stderr)
        print(f"INFO: Engine will use its internal defaults", file=sys.stderr)
    
    return engine_config

def get_function_parameters(func):
    """Get all parameter names that a function accepts"""
    try:
        sig = inspect.signature(func)
        return list(sig.parameters.keys())
    except (ValueError, TypeError):
        # Fallback for functions where signature inspection fails
        return []

def filter_params_for_function(func, params_dict, verbose=False):
    """
    Filter a parameters dictionary to only include parameters the function accepts
    """
    try:
        sig = inspect.signature(func)
        valid_params = set(sig.parameters.keys())
    except (ValueError, TypeError):
        # If we can't inspect the function, pass all parameters
        if verbose:
            print(f"WARNING: Cannot inspect function signature, passing all parameters", file=sys.stderr)
        return params_dict
    
    # Filter to only valid parameters with non-None values
    filtered = {}
    for k, v in params_dict.items():
        if k in valid_params and v is not None:
            filtered[k] = v
    
    if verbose:
        passed_params = list(filtered.keys())
        if passed_params:
            print(f"STATUS: Passing parameters to function: {', '.join(passed_params)}", file=sys.stderr)
        ignored_params = [k for k, v in params_dict.items() 
                         if k not in valid_params and v is not None]
        if ignored_params:
            print(f"STATUS: Ignoring parameters not accepted by function: {', '.join(ignored_params)}", file=sys.stderr)
    
    return filtered

def apply_dynamic_parameters(func, base_params, engine_config, verbose=False):
    """
    Dynamically apply engine config parameters to a function call
    Only applies parameters that the function actually accepts
    """
    # Start with base parameters
    final_params = base_params.copy()
    
    # Add any engine config parameters that the function accepts
    filtered_config = filter_params_for_function(func, engine_config, verbose=verbose)
    final_params.update(filtered_config)
    
    return final_params

def call_with_dynamic_params(func, base_params, engine_config, verbose=False):
    """
    Call a function with dynamically applied parameters
    """
    final_params = apply_dynamic_parameters(func, base_params, engine_config, verbose)
    return func(**final_params)

def merge_configs(*configs):
    """
    Merge multiple config dictionaries, with later ones taking precedence
    """
    result = {}
    for config in configs:
        if isinstance(config, dict):
            result.update(config)
    return result

def safe_get_nested_config(config, *keys, default=None):
    """
    Safely get nested configuration values
    Example: safe_get_nested_config(config, 'xtts', 'voice', 'style', default='neutral')
    """
    current = config
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current

def show_engine_config_summary(config, engine_name):
    """
    Display a summary of the engine's configuration
    """
    engine_config = extract_engine_config(config, engine_name)
    
    if not engine_config:
        print(f"📋 {engine_name.upper()}: No configuration section found", file=sys.stderr)
        return
    
    print(f"📋 {engine_name.upper()} Configuration Summary:", file=sys.stderr)
    
    # Show non-null parameters
    active_params = [(k, v) for k, v in engine_config.items() if v is not None]
    if active_params:
        for key, value in active_params:
            # Truncate long values for display
            if isinstance(value, str) and len(value) > 50:
                display_value = value[:47] + "..."
            elif isinstance(value, (list, dict)):
                display_value = f"{type(value).__name__} with {len(value)} items"
            else:
                display_value = value
            print(f"  {key}: {display_value}", file=sys.stderr)
    else:
        print(f"  No active parameters (all None/null)", file=sys.stderr)

# ============================================================================
# ENGINE PROCESSING WITH DYNAMIC CONFIG
# ============================================================================

def ensure_engine_config(config, engine_name):
    """REMOVED: No longer adds missing config sections
    
    The config file is the single source of truth.
    If a section is missing, the engine should handle it gracefully
    or the user should add it to their config file.
    """
    print(f"STATUS: Using {engine_name} config AS-IS from config file", file=sys.stderr)
    
    if engine_name not in config:
        print(f"WARNING: No {engine_name} section in config file", file=sys.stderr)
        print(f"INFO: Engine will use its own defaults for missing parameters", file=sys.stderr)
    else:
        # Show configuration summary
        show_engine_config_summary(config, engine_name)
    
    return False  # Never indicates config was updated

def process_with_engine(engine_name, text_file, output_dir, config, paths):
    """Process text file with the specified engine using dynamic configuration"""
    processor = get_engine_processor(engine_name)
    
    print(f"STATUS: Processing with {engine_name.upper()} engine", file=sys.stderr)
    print(f"STATUS: Config file is single source of truth", file=sys.stderr)
    
    # Show engine configuration summary
    show_engine_config_summary(config, engine_name)
    
    # Call the engine processor with config AS-IS
    return processor(text_file, output_dir, config, paths)

# ============================================================================
# CONVENIENCE FUNCTIONS FOR ENGINES
# ============================================================================

def get_engine_config_with_fallbacks(config, engine_name, fallback_config=None):
    """
    Get engine config with optional fallback defaults
    Useful for engines that want to provide internal defaults
    """
    engine_config = extract_engine_config(config, engine_name)
    
    if fallback_config:
        # Apply fallbacks for missing values
        result = fallback_config.copy()
        result.update({k: v for k, v in engine_config.items() if v is not None})
        return result
    
    return engine_config

def validate_required_params(engine_config, required_params, engine_name):
    """
    Validate that required parameters are present and not None
    Returns list of missing parameters
    """
    missing = []
    for param in required_params:
        if param not in engine_config or engine_config[param] is None:
            missing.append(param)
    
    if missing:
        print(f"ERROR: {engine_name} missing required parameters: {', '.join(missing)}", file=sys.stderr)
    
    return missing

def create_generation_params(base_params, engine_config, filter_function=None, verbose=False):
    """
    Create generation parameters by merging base params with engine config
    Optionally filter through a specific function's signature
    """
    result = base_params.copy()
    
    # Add non-null engine config parameters
    for k, v in engine_config.items():
        if v is not None:
            result[k] = v
    
    # Filter through function if provided
    if filter_function:
        result = filter_params_for_function(filter_function, result, verbose=verbose)
    
    return result

# ============================================================================
# ENGINE REGISTRATION
# ============================================================================

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