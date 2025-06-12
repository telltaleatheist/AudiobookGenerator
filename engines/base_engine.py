#!/usr/bin/env python3
"""
Base Engine - Abstract interface for TTS engines
All engines must implement this interface and use config_manager pattern
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any
from core.progress_display_manager import log_info
from managers.config_manager import ConfigManager, ConfigError

class BaseTTSEngine(ABC):
    """Abstract base class for TTS engines"""
    
    def __init__(self, engine_name: str):
        self.engine_name = engine_name
        self.config_manager = ConfigManager()
    
    @abstractmethod
    def process_text_file(self, text_file: str, output_dir: str, config: Dict[str, Any], paths: Dict[str, Any]) -> List[str]:
        """
        Process a text file and generate audio chunks
        
        Args:
            text_file: Path to input text file
            output_dir: Directory for output audio files
            config: Complete configuration dictionary
            paths: Pipeline paths dictionary
            
        Returns:
            List of generated audio file paths
            
        Raises:
            ConfigError: If required configuration is missing
        """
        pass
    
    def validate_required_config(self, engine_config: Dict[str, Any], required_keys: List[str]):
        """Validate that all required configuration keys are present"""
        missing_keys = []
        for key in required_keys:
            if key not in engine_config or engine_config[key] is None:
                missing_keys.append(key)
        
        if missing_keys:
            missing_str = ', '.join(missing_keys)
            raise ConfigError(f"Missing required {self.engine_name.upper()} configuration: {missing_str}")

def extract_engine_config(config: Dict[str, Any], engine_name: str, verbose: bool = False) -> Dict[str, Any]:
    """Extract engine-specific configuration with validation"""
    try:
        engine_config = config[engine_name]
        return engine_config
        
    except KeyError:
        from managers.config_manager import ConfigError
        raise ConfigError(f"Missing configuration section: {engine_name}")

def create_generation_params(base_params: Dict[str, Any], config: Dict[str, Any], 
                        filter_function=None, verbose: bool = False) -> Dict[str, Any]:
    """Create generation parameters from base params and config"""
    final_params = base_params.copy()
    
    # Add any additional config parameters that aren't in base_params
    for key, value in config.items():
        if key not in final_params and value is not None:
            final_params[key] = value
    
    # Filter for function if provided
    if filter_function:
        final_params = filter_params_for_function(final_params, filter_function, verbose)
    
    return final_params

def validate_required_params(config: Dict[str, Any], required_keys: List[str], engine_name: str) -> List[str]:
    """Validate required parameters and return list of missing ones"""
    missing_keys = []
    for key in required_keys:
        if key not in config or config[key] is None:
            missing_keys.append(key)
    
    return missing_keys

def filter_params_for_function(params: Dict[str, Any], func, verbose: bool = False) -> Dict[str, Any]:
    """Filter parameters to only include those accepted by a function"""
    import inspect
    
    try:
        sig = inspect.signature(func)
        valid_params = set(sig.parameters.keys())
        
        filtered = {k: v for k, v in params.items() if k in valid_params}
        
        if verbose:
            removed = set(params.keys()) - valid_params
            if removed:
                log_info("DEBUG: Filtered out unsupported params: {removed}")
        
        return filtered
        
    except Exception as e:
        if verbose:
            log_info("DEBUG: Could not filter parameters: {e}")
        return params

# Registry for available engines
_ENGINES = {}

def register_engine(name: str, processor_func):
    """Register a TTS engine processor function"""
    _ENGINES[name] = processor_func

def get_available_engines() -> List[str]:
    """Get list of available engine names"""
    return list(_ENGINES.keys())

def get_engine_processor(engine_name: str):
    """Get the processor function for an engine"""
    if engine_name not in _ENGINES:
        available = ', '.join(get_available_engines())
        raise ValueError(f"Unknown TTS engine: {engine_name}. Available: {available}")
    
    return _ENGINES[engine_name]

def ensure_engine_config(config: Dict[str, Any], engine_name: str) -> bool:
    """Validate engine configuration exists (no longer adds defaults)"""
    if engine_name not in config:
        raise ConfigError(f"Missing configuration section: {engine_name}")
    
    return False  # Never indicates config was updated

def process_with_engine(engine_name: str, text_file: str, output_dir: str, 
                       config: Dict[str, Any], paths: Dict[str, Any]) -> List[str]:
    """Process text file with the specified engine"""
    processor = get_engine_processor(engine_name)
    
    # Validate engine config exists
    ensure_engine_config(config, engine_name)
    
    # Call the engine processor
    return processor(text_file, output_dir, config, paths)
