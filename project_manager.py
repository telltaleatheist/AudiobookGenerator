#!/usr/bin/env python3
"""
Project Manager - Clean project and configuration management
Handles project creation, source files, batch naming, and config inheritance
Updated with automatic voice sample pairing for F5/XTTS
"""

import os
import json
import shutil
import re
from pathlib import Path
from datetime import datetime

class ProjectManager:
    """Manages audiobook projects and configurations"""
    
    def __init__(self, base_dir="output"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
    
    def create_project(self, name):
        """Create a new project with basic structure"""
        if not re.match(r'^[a-zA-Z0-9_-]+$', name):
            raise ValueError(f"Invalid project name: {name}")
        
        project_dir = self.base_dir / name
        if project_dir.exists():
            raise ValueError(f"Project '{name}' already exists")
        
        # Create structure
        project_dir.mkdir()
        (project_dir / "source").mkdir()
        (project_dir / "config").mkdir()
        (project_dir / "jobs").mkdir()
        (project_dir / "samples").mkdir()
        
        # Create README
        readme = f"""# Audiobook Project: {name}

## Usage
- Place source files (.epub, .txt) in `source/`
- Place voice samples (.wav, .mp3) in `samples/` for voice cloning
- Run: `python AudiobookGenerator.py --project {name}`
- Process sections: `python AudiobookGenerator.py --project {name} --sections 1 2 3`

## Structure
- `source/` - Input files
- `samples/` - Voice reference audio files for cloning
- `config/` - Batch configurations
- `jobs/` - Processing outputs (batches/sections)
"""
        
        (project_dir / "README.md").write_text(readme)
        
        print(f"✅ Created project '{name}' at {project_dir}")
        return project_dir
    
    def validate_project(self, name):
        """Ensure project exists and has proper structure"""
        project_dir = self.base_dir / name
        if not project_dir.exists():
            raise ValueError(f"Project '{name}' not found")
        
        # Ensure required directories exist
        (project_dir / "source").mkdir(exist_ok=True)
        (project_dir / "config").mkdir(exist_ok=True)
        (project_dir / "samples").mkdir(exist_ok=True)
        
        return project_dir
    
    def find_source_file(self, project_name):
        """Find source file in project"""
        project_dir = self.validate_project(project_name)
        source_dir = project_dir / "source"
        
        files = list(source_dir.glob("*.epub")) + list(source_dir.glob("*.txt"))
        if not files:
            raise ValueError(f"No source files in {source_dir}")
        
        if len(files) == 1:
            return files[0]
        
        # Multiple files - let user choose
        print("Multiple source files found:")
        for i, f in enumerate(files, 1):
            print(f"  {i}. {f.name}")
        
        while True:
            try:
                choice = int(input("Choose file (number): ")) - 1
                if 0 <= choice < len(files):
                    return files[choice]
            except (ValueError, KeyboardInterrupt):
                raise ValueError("No file selected")
    
    def add_source_file(self, project_name, input_file):
        """Copy source file to project"""
        project_dir = self.validate_project(project_name)
        source_dir = project_dir / "source"
        
        input_path = Path(input_file)
        if not input_path.exists():
            raise ValueError(f"File not found: {input_file}")
        
        if input_path.suffix.lower() not in ['.epub', '.txt']:
            raise ValueError(f"Unsupported file type: {input_path.suffix}")
        
        dest = source_dir / input_path.name
        
        # Handle name conflicts
        counter = 2
        while dest.exists():
            stem = input_path.stem
            suffix = input_path.suffix
            dest = source_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        
        shutil.copy2(input_path, dest)
        print(f"📁 Added source file: {dest.name}")
        return dest
    
    def find_voice_samples(self, project_name):
        """Find all voice sample files in project samples directory"""
        project_dir = self.validate_project(project_name)
        samples_dir = project_dir / "samples"
        
        # Supported audio formats
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        
        samples = []
        for ext in audio_extensions:
            samples.extend(list(samples_dir.glob(f"*{ext}")))
            samples.extend(list(samples_dir.glob(f"*{ext.upper()}")))
        
        # Sort by name for consistent ordering
        samples = sorted(set(samples), key=lambda p: p.name.lower())
        
        return samples
    
    def get_voice_sample_pairs(self, project_name):
        """Get voice sample pairs (audio + text) for F5-TTS"""
        project_dir = self.validate_project(project_name)
        samples_dir = project_dir / "samples"
        
        # Find audio files (avoid duplicates)
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        audio_files = set()  # Use set to avoid duplicates
        
        for ext in audio_extensions:
            # Only check lowercase to avoid duplicates
            audio_files.update(samples_dir.glob(f"*{ext}"))
        
        # Convert back to sorted list
        audio_files = sorted(audio_files, key=lambda p: p.name.lower())
        
        # Find matching text files for each audio file
        sample_pairs = []
        for audio_file in audio_files:
            # Look for matching .txt file with same stem
            text_file = samples_dir / f"{audio_file.stem}.txt"
            if text_file.exists():
                # Read the text content
                try:
                    with open(text_file, 'r', encoding='utf-8') as f:
                        ref_text = f.read().strip()
                    sample_pairs.append({
                        'audio': audio_file,
                        'text': text_file,
                        'ref_text': ref_text
                    })
                except Exception as e:
                    print(f"⚠️ Could not read {text_file.name}: {e}")
            else:
                print(f"⚠️ No matching .txt file found for {audio_file.name}")
        
        return sample_pairs
    
    def get_primary_voice_sample_pair(self, project_name):
        """Get the primary voice sample pair for F5-TTS"""
        pairs = self.get_voice_sample_pairs(project_name)
        
        if not pairs:
            return None
        
        if len(pairs) == 1:
            pair = pairs[0]
            print(f"🎤 F5: Using {pair['audio'].name} + {pair['text'].name}")
            return pair
        
        # Multiple pairs - show options and use first
        print(f"🎤 Found {len(pairs)} voice sample pairs:")
        for i, pair in enumerate(pairs, 1):
            print(f"  {i}. {pair['audio'].name} + {pair['text'].name}")
        
        print(f"🎤 Auto-selecting: {pairs[0]['audio'].name}")
        return pairs[0]
    
    def get_all_voice_samples(self, project_name):
        """Get all voice samples for engines that support multiple references"""
        samples = self.find_voice_samples(project_name)
        
        if samples:
            print(f"🎤 Found {len(samples)} voice samples: {', '.join(s.name for s in samples)}")
        
        return samples
    
    def add_voice_sample(self, project_name, sample_file):
        """Copy voice sample to project samples directory"""
        project_dir = self.validate_project(project_name)
        samples_dir = project_dir / "samples"
        
        sample_path = Path(sample_file)
        if not sample_path.exists():
            raise ValueError(f"Sample file not found: {sample_file}")
        
        # Check if it's an audio file
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        if sample_path.suffix.lower() not in audio_extensions:
            raise ValueError(f"Unsupported audio format: {sample_path.suffix}")
        
        dest = samples_dir / sample_path.name
        
        # Handle name conflicts
        counter = 2
        while dest.exists():
            stem = sample_path.stem
            suffix = sample_path.suffix
            dest = samples_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        
        shutil.copy2(sample_path, dest)
        print(f"🎤 Added voice sample: {dest.name}")
        return dest
    
    def get_batch_name(self, project_name, sections=None):
        """Generate batch name based on sections"""
        project_dir = self.validate_project(project_name)
        
        if sections:
            sections = sorted(set(sections))
            if len(sections) == 1:
                base = f"batch{sections[0]}"
            else:
                # Create range notation: [1,2,3,5,6] -> "batch1-3_5-6"
                ranges = []
                start = sections[0]
                end = sections[0]
                
                for i in range(1, len(sections)):
                    if sections[i] == end + 1:
                        end = sections[i]
                    else:
                        ranges.append(f"{start}-{end}" if start != end else str(start))
                        start = end = sections[i]
                
                ranges.append(f"{start}-{end}" if start != end else str(start))
                base = f"batch{'_'.join(ranges)}"
        else:
            base = "complete"
        
        # Handle collisions
        batch_name = base
        counter = 2
        while (project_dir / batch_name).exists():
            batch_name = f"{base}_{counter}"
            counter += 1
        
        return batch_name
    
    def get_batch_paths(self, project_name, batch_name, tts_engine):
        """Get all paths for a batch"""
        project_dir = self.validate_project(project_name)
        batch_dir = project_dir / "jobs" / batch_name
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        engine_suffix = "_edge" if tts_engine == 'edge' else "_bark"
        
        paths = {
            'project_dir': project_dir,
            'batch_dir': batch_dir,
            'temp_dir': batch_dir / "temp_files",
            'chunks': batch_dir / f"{project_name}_{batch_name}_chunks{engine_suffix}.txt",
            'combined': batch_dir / f"{project_name}_{batch_name}{engine_suffix}_combined.wav",
            'final': batch_dir / f"{project_name}_{batch_name}.wav",
            'log': batch_dir / "progress.log",
            'config': project_dir / "config" / "config.json",  # Single config file
            'job_config': batch_dir / "config.json"  # Copy in job directory
        }
        
        paths['temp_dir'].mkdir(exist_ok=True)
        return paths
    
    def get_default_config(self, tts_engine):
        """Get default configuration for engine"""
        return {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'tts_engine': tts_engine
            },
            'bark': {
                'voice': 'v2/en_speaker_0',
                'text_temp': 0.1,
                'waveform_temp': 0.15
            },
            'edge': {
                'voice': 'en-US-AriaNeural',
                'rate': '+0%',
                'pitch': '+0Hz',
                'volume': '+0%',
                'delay': 1.5,
                'use_ssml': False
            },
            'f5': {
                'model_type': 'F5-TTS',
                'model_name': 'F5TTS_Base',
                'ref_audio': None,
                'ref_text': None,
                'chunk_max_chars': 300,
                'target_chars': 200,
                'speed': 1.0,
                'sample_rate': 24000
            },
            'xtts': {
                'model_name': 'tts_models/multilingual/multi-dataset/xtts_v2',
                'language': 'en',
                'speaker': None,
                'speaker_wav': None,
                'chunk_max_chars': 400,
                'target_chars': 300,
                'speed': 1.0,
                'temperature': 0.75,
                'gpu_acceleration': True
            },
            'rvc': {
                'model': 'Sigma Male Narrator',
                'speed_factor': 1.0,
                'clean_silence': True,
                'silence_threshold': -40.0,
                'silence_duration': 0.6
            },
            'audio': {
                'silence_gap': 0.3
            },
            'pipeline': {
                'cleanup_temp_files': True,
                'cleanup_intermediate_files': True
            }
        }
    
    def create_config(self, project_name, batch_name, tts_engine, sections=None, 
                        source_file=None, inherit_from=None, cli_overrides=None):
            """Create configuration for a batch with automatic voice sample detection"""
            config = self.get_default_config(tts_engine)
            
            # Update metadata
            config['metadata'].update({
                'batch_name': batch_name,
                'project_name': project_name,
                'tts_engine': tts_engine,
                'sections': sections,
                'source_file': str(source_file) if source_file else None
            })
            
            # Try to inherit from existing project config
            project_config_path = self.validate_project(project_name) / "config" / "config.json"
            if project_config_path.exists():
                try:
                    parent_config = self.load_config(project_config_path)
                    config = self._merge_configs(config, parent_config)
                    config['metadata']['inherited_from'] = "config.json"
                    print(f"📄 Inherited from: config.json")
                except Exception as e:
                    print(f"⚠️ Could not inherit from config.json: {e}")
            
            # Auto-detect and add voice samples AFTER inheritance (so it overrides inherited values)
            if tts_engine in ['f5', 'xtts']:
                if tts_engine == 'f5':
                    voice_samples = self.find_voice_samples(project_name)
                    if voice_samples:
                        config['f5']['ref_audio'] = str(voice_samples[0])
                        config['f5']['ref_text'] = ""  # Always empty for auto-transcription
                        print(f"🎤 F5: Using {voice_samples[0].name} (auto-transcribe)")
                    else:
                        print(f"ℹ️ No voice samples found for F5")
                        print(f"💡 Add .wav files to samples/ directory")
                
                elif tts_engine == 'xtts':
                    # XTTS uses only audio files (ignores .txt files)
                    voice_samples = self.find_voice_samples(project_name)
                    if voice_samples:
                        all_samples = self.get_all_voice_samples(project_name)
                        if all_samples:
                            if len(all_samples) == 1:
                                config['xtts']['speaker_wav'] = str(all_samples[0])
                            else:
                                # Store multiple samples as a list (XTTS supports this)
                                config['xtts']['speaker_wav'] = [str(s) for s in all_samples]
                            print(f"🎤 XTTS: Using {len(all_samples)} audio sample(s)")
                    else:
                        print(f"ℹ️ No voice samples found for XTTS")
                        print(f"💡 Add .wav files to samples/ directory")
            else:
                print(f"ℹ️ {tts_engine.upper()} doesn't use voice samples")
            
            # Apply CLI overrides LAST (so they override everything)
            if cli_overrides:
                for section, overrides in cli_overrides.items():
                    if section in config:
                        config[section].update(overrides)
            
            return config
    
    def save_config(self, config, config_path):
        """Save configuration to file"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_config(self, config_path):
        """Load configuration from file"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def find_config(self, project_name, config_spec):
        """Find config file by name or path"""
        project_dir = self.validate_project(project_name)
        config_dir = project_dir / "config"
        
        # Try as direct path
        config_path = Path(config_spec)
        if config_path.is_absolute() and config_path.exists():
            return config_path
        
        # Try relative to project
        project_relative = project_dir / config_spec
        if project_relative.exists():
            return project_relative
        
        # Try in config directory
        config_file = config_dir / config_spec
        if config_file.exists():
            return config_file
        
        # Try with .json extension
        if not config_spec.endswith('.json'):
            config_file = config_dir / f"{config_spec}.json"
            if config_file.exists():
                return config_file
        
        available = [f.name for f in config_dir.glob("*.json")]
        if available:
            raise ValueError(f"Config '{config_spec}' not found. Available: {', '.join(available)}")
        else:
            raise ValueError(f"Config '{config_spec}' not found. No configs exist yet.")
    
    def get_most_recent_config(self, project_name):
        """Get most recently created config"""
        project_dir = self.validate_project(project_name)
        config_dir = project_dir / "config"
        
        configs = list(config_dir.glob("*.json"))
        if not configs:
            return None
        
        return max(configs, key=lambda p: p.stat().st_mtime)
    
    def _merge_configs(self, base, parent):
        """Deep merge parent config into base, preserving base metadata"""
        base_metadata = base.get('metadata', {})
        
        for section, values in parent.items():
            if section == 'metadata':
                continue  # Don't inherit metadata
            if section in base and isinstance(values, dict):
                base[section].update(values)
        
        base['metadata'] = base_metadata
        return base
    
    def display_config_summary(self, config):
        """Display configuration summary"""
        print(f"🎛️ Configuration Summary")
        print(f"📋 Batch: {config['metadata']['batch_name']}")
        print(f"🎤 TTS Engine: {config['metadata']['tts_engine'].upper()}")
        
        if config['metadata'].get('inherited_from'):
            print(f"📄 Inherited from: {config['metadata']['inherited_from']}")
        
        engine = config['metadata']['tts_engine']
        if engine == 'edge':
            edge = config['edge']
            print(f"🎙️ Voice: {edge['voice']}")
            print(f"⚡ Rate: {edge['rate']}, Pitch: {edge['pitch']}")
        else:
            bark = config['bark']
            print(f"🎙️ Voice: {bark['voice']}")
            print(f"🌡️ Temps: text={bark['text_temp']}, waveform={bark['waveform_temp']}")
        
        print(f"🎭 RVC Model: {config['rvc']['model']}")
        print(f"⚡ Speed: {config['rvc']['speed_factor']}x")
        
        if config['metadata'].get('sections'):
            sections = ', '.join(map(str, config['metadata']['sections']))
            print(f"🎯 Sections: {sections}")