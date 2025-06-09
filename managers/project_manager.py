#!/usr/bin/env python3
"""
Project Manager - Clean project and configuration management
FIXED: Added proper imports and cleaned up logging calls
"""

import os
import json
import shutil
import re
from pathlib import Path
from datetime import datetime
import sys
from core.progress_display_manager import log_info, log_success, log_warning

class ProjectManager:
    """Manages audiobook projects and configurations"""
    
    def __init__(self, base_dir="output"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Path to default config file
        self.default_config_path = Path("default_config.json")
        if not self.default_config_path.exists():
            # Fallback to look in script directory
            script_dir = Path(__file__).parent
            self.default_config_path = script_dir / "default_config.json"
    
    def create_project(self, name):
        """Create a new project with basic structure and copy default config"""
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
        
        # Copy default config file
        project_config_path = project_dir / "config" / "config.json"
        if self.default_config_path.exists():
            shutil.copy2(self.default_config_path, project_config_path)
            
            # Update metadata in the copied config
            self._update_project_metadata(project_config_path, name)
        else:
            log_warning(f"Default config not found at {self.default_config_path}")
            log_info("💡 Please create default_config.json in the project root")
            raise FileNotFoundError(f"Default config file not found: {self.default_config_path}")
        
        # Create README
        readme = f"""# Audiobook Project: {name}

## Usage
- Place source files (.epub, .pdf, .txt) in `source/`
- Place voice samples (.wav, .mp3) in `samples/` for voice cloning
- Run: `python AudiobookGenerator.py --project {name}`
- Process sections: `python AudiobookGenerator.py --project {name} --sections 1 2 3`

## Configuration
- Edit `config/config.json` to customize TTS and RVC settings
- The config file contains all available parameters for each engine

## Structure
- `source/` - Input files (.epub, .pdf, .txt)
- `samples/` - Voice reference audio files for cloning
- `config/` - Project configuration (copied from default_config.json)
- `jobs/` - Processing outputs (batches/sections)
"""
        
        (project_dir / "README.md").write_text(readme)
        
        log_success(f"Created project '{name}' at {project_dir}")
        return project_dir
    
    def _update_project_metadata(self, config_path, project_name):
        """Update metadata in the copied config file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Update metadata
            if 'metadata' not in config:
                config['metadata'] = {}
            
            config['metadata'].update({
                'created_at': datetime.now().isoformat(),
                'project_name': project_name,
                'config_source': 'default_config.json'
            })
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
        except Exception as e:
            log_warning(f"Could not update config metadata: {e}")
    
    def validate_project(self, name):
        """Ensure project exists and has proper structure"""
        project_dir = self.base_dir / name
        if not project_dir.exists():
            raise ValueError(f"Project '{name}' not found")
        
        # Ensure required directories exist
        (project_dir / "source").mkdir(exist_ok=True)
        (project_dir / "config").mkdir(exist_ok=True)
        (project_dir / "samples").mkdir(exist_ok=True)
        
        # Ensure config file exists
        config_path = project_dir / "config" / "config.json"
        if not config_path.exists():
            if self.default_config_path.exists():
                shutil.copy2(self.default_config_path, config_path)
                self._update_project_metadata(config_path, name)
            else:
                raise FileNotFoundError(f"No config file and no default config found")
        
        return project_dir
    
    def find_source_file(self, project_name):
        """Find source file in project"""
        project_dir = self.validate_project(project_name)
        source_dir = project_dir / "source"
        
        # Look for files with both lowercase and uppercase extensions
        files = []
        for ext in ['.epub', '.txt', '.pdf']:
            files.extend(list(source_dir.glob(f"*{ext}")))
            files.extend(list(source_dir.glob(f"*{ext.upper()}")))
        
        # Remove duplicates while preserving order
        unique_files = []
        seen = set()
        for f in files:
            if f not in seen:
                unique_files.append(f)
                seen.add(f)
        files = unique_files
        
        if not files:
            raise ValueError(f"No source files in {source_dir}")
        
        if len(files) == 1:
            return files[0]
        
        # Multiple files - let user choose
        log_info("Multiple source files found:")
        for i, f in enumerate(files, 1):
            log_info(f"  {i}. {f.name}")
        
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
        
        # Check file extension (case insensitive)
        if input_path.suffix.lower() not in ['.epub', '.txt', '.pdf']:
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
                    log_warning(f"Could not read {text_file.name}: {e}")
            else:
                log_warning(f"No matching .txt file found for {audio_file.name}")
        
        return sample_pairs
    
    def get_primary_voice_sample_pair(self, project_name):
        """Get the primary voice sample pair for F5-TTS"""
        pairs = self.get_voice_sample_pairs(project_name)
        
        if not pairs:
            return None
        
        if len(pairs) == 1:
            pair = pairs[0]
            return pair
        
        # Multiple pairs - use first
        return pairs[0]
    
    def get_all_voice_samples(self, project_name):
        """Get all voice samples for engines that support multiple references"""
        samples = self.find_voice_samples(project_name)
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
        """Get all paths for a batch - CLEANED UP"""
        project_dir = self.validate_project(project_name)
        batch_dir = project_dir / "jobs" / batch_name
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        paths = {
            'project_dir': project_dir,
            'batch_dir': batch_dir,
            'temp_dir': batch_dir / "temp_files",
            'sections_dir': batch_dir / "sections",
            'final': batch_dir / f"{project_name}_{batch_name}_master.mp3",
            'log': batch_dir / "progress.log",
            'config': project_dir / "config" / "config.json",
            'job_config': batch_dir / "config.json"
        }
        
        # Create required directories
        paths['sections_dir'].mkdir(exist_ok=True)
        return paths
    
    def create_config(self, project_name, batch_name, tts_engine, sections=None, 
                            source_file=None, inherit_from=None, cli_overrides=None):
        """Load and update existing config (NO MORE CONFIG GENERATION)"""
        
        # Load the existing project config
        project_config_path = self.validate_project(project_name) / "config" / "config.json"
        
        try:
            config = self.load_config(project_config_path)
        except Exception as e:
            raise ValueError(f"Could not load project config: {e}")
        
        # Update metadata for this specific job
        if 'metadata' not in config:
            config['metadata'] = {}
            
        config['metadata'].update({
            'last_accessed': datetime.now().isoformat(),
            'batch_name': batch_name,
            'tts_engine': tts_engine,
            'sections': sections,
            'source_file': str(source_file) if source_file else None,
            'last_batch': batch_name,
            'project_name': project_name
        })

        # Auto-detect and add voice samples AFTER loading config
        if tts_engine in ['f5', 'xtts']:
            if tts_engine == 'f5':
                voice_samples = self.find_voice_samples(project_name)
                if voice_samples:
                    config['f5']['ref_audio'] = str(voice_samples[0])
                    config['f5']['ref_text'] = ""  # Always empty for auto-transcription
                else:
                    # CLEANED: Remove verbose message
                    pass
            
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
                else:
                    # CLEANED: Remove verbose message
                    pass
        
        # Apply CLI overrides LAST (so they override everything)
        if cli_overrides:
            for section, overrides in cli_overrides.items():
                if section in config:
                    config[section].update(overrides)
                else:
                    # If the section doesn't exist, create it
                    config[section] = overrides
        
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
    
    def display_config_summary(self, config):
        """CLEANED: Display minimal configuration summary"""
        # REMOVED: All verbose config display
        # This information will be in the config file and progress.log if needed
        pass
    
    def update_config_timestamp(self, config_path, batch_name):
        """Update only the timestamp and last batch info in main config"""
        config_path = Path(config_path)
        
        if config_path.exists():
            try:
                # Load existing config
                with open(config_path, 'r') as f:
                    existing_config = json.load(f)
                
                # Update only metadata timestamps, don't touch anything else
                if 'metadata' not in existing_config:
                    existing_config['metadata'] = {}
                
                existing_config['metadata']['last_accessed'] = datetime.now().isoformat()
                existing_config['metadata']['last_batch'] = batch_name
                
                # Save back with minimal changes
                with open(config_path, 'w') as f:
                    json.dump(existing_config, f, indent=2)
                
            except Exception as e:
                # CLEANED: Log to progress instead of console
                pass
        else:
            # CLEANED: Log to progress instead of console
            pass
