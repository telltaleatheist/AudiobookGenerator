#!/usr/bin/env python3
"""
Config Manager - Centralized configuration system with no code defaults
All defaults come from default_config.json - if missing values are found at runtime, 
the system fails gracefully with helpful error messages.
"""

import json
import shutil
import sys
from pathlib import Path
from datetime import datetime

class ConfigError(Exception):
    """Custom exception for configuration errors"""
    pass

class ConfigManager:
    """Manages configuration files with no hardcoded defaults"""
    
    def __init__(self, script_root=None):
        """Initialize config manager
        
        Args:
            script_root: Path to script root directory (auto-detected if None)
        """
        if script_root is None:
            # Auto-detect script root (where this file is located)
            self.script_root = Path(__file__).parent
        else:
            self.script_root = Path(script_root)
        
        self.default_config_path = self.script_root / "default_config.json"
    
    def ensure_default_config_exists(self):
        """Create default_config.json if it doesn't exist"""
        if self.default_config_path.exists():
            print(f"✅ Default config found: {self.default_config_path}")
            return True
        
        print(f"📄 Creating default config: {self.default_config_path}")
        
        # Create comprehensive default configuration
        default_config = self._create_default_config()
        
        try:
            with open(self.default_config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2)
            
            print(f"✅ Default config created successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create default config: {e}")
            return False
    
    def _create_default_config(self):
        """Create the default configuration with default metadata and current settings"""
        return {
            "metadata": {
                "config_version": "2.0",
                "config_type": "default",
                "created_at": datetime.now().isoformat(),
                "description": "Default configuration for AudiobookGenerator with dynamic parameter loading",
                "rvc_voice": "sigma_male_narrator"
            },
            "bark": {
                "voice": "v2/en_speaker_0",
                "history_prompt": None,
                "text_temp": 0.1,
                "waveform_temp": 0.15,
                "silent": False,
                "chunk_max_chars": 150,
                "target_chars": 120,
                "use_smaller_models": False,
                "reload_model_every_chunks": 15,
                "reload_model_every_chars": 2000,
                "clear_cuda_cache": True,
                "force_cpu": False,
                "offload_cpu": False,
                "normalize_audio": True,
                "trim_silence": True,
                "fade_in": 0.0,
                "fade_out": 0.05,
                "detect_artifacts": True,
                "trim_artifacts": True,
                "artifact_threshold": 2.5,
                "silence_threshold": 0.01,
                "repetition_detection": True,
                "max_duration_per_char": 0.08,
                "seed": None,
                "randomize_seed_per_chunk": False,
                "retry_failed_chunks": 3,
                "skip_failed_chunks": False,
                "error_recovery_mode": "retry",
                "output_format": "wav",
                "bit_depth": 16,
                "verbose": False,
                "debug_output": False,
                "post_process_audio": True
            },
            "edge": {
                "voice": "en-US-AriaNeural",
                "rate": "+0%",
                "pitch": "+0Hz",
                "volume": "+0%",
                "chunk_max_chars": 1000,
                "target_chars": 800,
                "delay": 1.5,
                "normalize_text": True,
                "expand_abbreviations": True,
                "spell_out_numbers": False,
                "streaming": False,
                "retry_attempts": 3,
                "retry_delay": 2.0,
                "fallback_voice": None,
                "ignore_errors": False,
                "skip_failed_chunks": False,
                "verbose": False,
                "debug_output": False
            },
            "f5": {
                "model_type": "F5-TTS",
                "model_name": "F5TTS_Base",
                "ref_audio": None,
                "chunk_max_chars": 350,
                "target_chars": 280,
                "speed": 0.9,
                "sample_rate": 24000,
                "cross_fade_duration": 0.12,
                "sway_sampling_coef": -0.8,
                "cfg_strength": 1.5,
                "nfe_step": 128,
                "seed": 42,
                "fix_duration": None,
                "remove_silence": False,
                "ref_text": ""
            },
            "xtts": {
                "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
                "language": "en",
                "speaker_wav": "null",
                "chunk_max_chars": 249,
                "target_chars": 140,
                "reload_model_every_chunks": 5,
                "speed": 1.0,
                "temperature": 0.5,
                "length_penalty": 0.95,
                "repetition_penalty": 7.0,
                "top_k": 15,
                "top_p": 0.6,
                "do_sample": True,
                "num_beams": 1,
                "enable_text_splitting": True,
                "gpt_cond_len": 12,
                "gpt_cond_chunk_len": 2,
                "max_ref_len": 20,
                "sound_norm_refs": True,
                "sample_rate": 24000,
                "normalize_audio": True,
                "retry_attempts": 5,
                "retry_delay": 2.0,
                "ignore_errors": False,
                "skip_failed_chunks": False,
                "verbose": True,
                "debug": True,
                "save_intermediate": False,
                "silence_gap_sentence": 0.3,
                "silence_gap_dramatic": 0.45,
                "silence_gap_paragraph": 0.45,
                "reset_state_between_chunks": True
            },
            "rvc_global": {
                "speed_factor": 1.0,
                "clean_silence": False,
                "silence_threshold": -40.0,
                "silence_duration": 0.6,
                "f0_method": "crepe",
                "hop_length": 64,
                "clean_voice": True,
                "clean_strength": 0.3,
                "autotune_voice": True
            },
            "rvc_my_voice": {
                "model": "my_voice",
                "n_semitones": -2,
                "index_rate": 0.35,
                "protect_rate": 0.15,
                "rms_mix_rate": 0.4,
                "split_voice": True,
                "autotune_strength": 0.05
            },
            "rvc_sigma_male_narrator": {
                "model": "Sigma Male Narrator",
                "n_semitones": -4,
                "index_rate": 0.4,
                "protect_rate": 0.4,
                "rms_mix_rate": 0.5,
                "split_voice": True,
                "autotune_strength": 0.3
            },
            "audio": {
                "silence_gap": 0.3,
                "gap_multiplier": 0.5,
                "min_gap": 0.3,
                "max_gap": 1.0,
                "analysis_duration": 0.8
            },
            "pipeline": {
                "cleanup_temp_files": True,
                "cleanup_intermediate_files": True
            },
            "openai": {
                "voice": "onyx",
                "api_key": None,
                "model": "tts-1",
                "chunk_max_chars": 4000,
                "retry_attempts": 3,
                "retry_delay": 1.0,
                "ignore_errors": False,
                "skip_failed_chunks": False
            }
        }
    
    def copy_default_to_project(self, project_dir, force_overwrite=False):
        """Copy default config to project directory
        
        Args:
            project_dir: Path to project directory
            force_overwrite: Whether to overwrite existing config
            
        Returns:
            Path to the created config file
        """
        project_dir = Path(project_dir)
        config_dir = project_dir / "config"
        config_file = config_dir / "config.json"
        
        # Ensure config directory exists
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if config already exists
        if config_file.exists() and not force_overwrite:
            print(f"✅ Project config already exists: {config_file}")
            return config_file
        
        # Ensure default config exists
        if not self.ensure_default_config_exists():
            raise ConfigError("Cannot create default config file")
        
        try:
            # Copy default to project
            shutil.copy2(self.default_config_path, config_file)
            
            # Update metadata for this project
            self._update_project_metadata(config_file, project_dir.name)
            
            action = "Overwritten" if config_file.exists() else "Created"
            print(f"📄 {action} project config: {config_file}")
            return config_file
            
        except Exception as e:
            raise ConfigError(f"Failed to copy config to project: {e}")
    
    def _update_project_metadata(self, config_file, project_name):
        """Update metadata in the project config file"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Update metadata
            if 'metadata' not in config:
                config['metadata'] = {}
            
            config['metadata'].update({
                'project_name': project_name,
                'config_source': 'default_config.json',
                'copied_at': datetime.now().isoformat(),
                'config_type': 'project_config'
            })
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Could not update project metadata: {e}")
    
    def load_config(self, config_file):
        """Load configuration from file with validation
        
        Args:
            config_file: Path to config file
            
        Returns:
            dict: Configuration data
            
        Raises:
            ConfigError: If config cannot be loaded
        """
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise ConfigError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            print(f"✅ Loaded config: {config_path}")
            return config
            
        except json.JSONDecodeError as e:
            raise ConfigError(f"Invalid JSON in config file: {config_path} - {e}")
        except Exception as e:
            raise ConfigError(f"Failed to load config: {config_path} - {e}")
    
    def get_config_value(self, config, key_path, required=True):
        """Get configuration value with path notation (e.g., 'xtts.temperature')
        
        Args:
            config: Configuration dictionary
            key_path: Dot-separated path to config value (e.g., 'xtts.temperature')
            required: Whether this value is required
            
        Returns:
            Configuration value
            
        Raises:
            ConfigError: If required value is missing
        """
        keys = key_path.split('.')
        current = config
        
        try:
            for key in keys:
                current = current[key]
            
            return current
            
        except (KeyError, TypeError):
            if required:
                raise ConfigError(f"Missing required configuration value: {key_path}")
            return None
    
    def validate_engine_config(self, config, engine_name, required_keys=None):
        """Validate that engine configuration has all required keys
        
        Args:
            config: Configuration dictionary
            engine_name: Name of the engine (e.g., 'xtts', 'bark')
            required_keys: List of required configuration keys
            
        Raises:
            ConfigError: If required configuration is missing
        """
        if engine_name not in config:
            raise ConfigError(f"Missing configuration section: {engine_name}")
        
        engine_config = config[engine_name]
        
        if required_keys:
            missing_keys = []
            for key in required_keys:
                if key not in engine_config or engine_config[key] is None:
                    missing_keys.append(key)
            
            if missing_keys:
                missing_str = ', '.join(missing_keys)
                raise ConfigError(f"Missing required {engine_name} configuration: {missing_str}")
        
        print(f"✅ {engine_name.upper()} configuration validated")
        return engine_config
    
    def create_project_config(self, project_name, output_dir="output"):
        """Create a new project with configuration
        
        Args:
            project_name: Name of the project
            output_dir: Base output directory
            
        Returns:
            Path to the created config file
        """
        project_dir = Path(output_dir) / project_name
        
        # Create project structure
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "source").mkdir(exist_ok=True)
        (project_dir / "samples").mkdir(exist_ok=True)
        (project_dir / "jobs").mkdir(exist_ok=True)
        
        # Copy config
        config_file = self.copy_default_to_project(project_dir)
        
        print(f"🚀 Created project '{project_name}' with configuration")
        return config_file

def main():
    """Command-line interface for config management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Configuration Manager")
    parser.add_argument("--create-default", action="store_true", 
                       help="Create default_config.json if missing")
    parser.add_argument("--create-project", help="Create new project with config")
    parser.add_argument("--copy-to-project", help="Copy default config to existing project")
    parser.add_argument("--validate", help="Validate configuration file")
    parser.add_argument("--output-dir", default="output", help="Output directory for projects")
    
    args = parser.parse_args()
    
    config_manager = ConfigManager()
    
    try:
        if args.create_default:
            config_manager.ensure_default_config_exists()
        
        elif args.create_project:
            config_manager.create_project_config(args.create_project, args.output_dir)
        
        elif args.copy_to_project:
            project_dir = Path(args.output_dir) / args.copy_to_project
            config_manager.copy_default_to_project(project_dir, force_overwrite=True)
        
        elif args.validate:
            config = config_manager.load_config(args.validate)
            print(f"✅ Configuration file is valid")
            
            # Show available engines
            engines = [k for k in config.keys() if k in ['bark', 'edge', 'f5', 'xtts', 'openai']]
            print(f"📋 Available engines: {', '.join(engines)}")
        
        else:
            parser.print_help()
    
    except ConfigError as e:
        print(f"❌ Configuration Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()