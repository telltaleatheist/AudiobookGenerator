#!/usr/bin/env python3
"""
Pipeline Manager - Section-based processing with resumable progress
Orchestrates TTS → RVC → Combine cycle for each section
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

class PipelineManager:
    """Manages section-based audiobook pipeline with resume capability"""
    
    def __init__(self):
        self.progress = {}
        self.progress_file = None
    
    def run(self, source_file: str, paths: Dict[str, Path], config: Dict[str, Any], 
            sections: Optional[List[int]] = None, skip_rvc: bool = False) -> bool:
        """Execute the section-based pipeline"""
        
        self.progress_file = paths['log']
        
        try:
            # Load or initialize progress
            self.progress = self._load_progress()
            
            # Initialize if starting fresh
            if 'status' not in self.progress:
                self._initialize_progress(source_file, paths, config, sections, skip_rvc)
            
            print(f"🚀 Section-based Pipeline")
            print(f"📁 Project: {config['metadata']['project_name']}")
            print(f"🎤 Engine: {config['metadata']['tts_engine'].upper()}")
            print(f"🎭 RVC: {config['metadata']['rvc_voice']}")
            
            # Phase 1: Preprocessing and section creation
            if not self._is_phase_complete('preprocessing'):
                if not self._run_preprocessing():
                    return False
            
            # Phase 2: Section processing loop
            if not self._run_section_processing():
                return False
            
            # Phase 3: Cleanup
            self._run_cleanup()
            
            print(f"\n🎉 Pipeline completed successfully!")
            print(f"🎵 Final audio: {self.progress['paths']['final']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Pipeline error: {e}")
            self._update_progress({'error': str(e), 'failed_at': datetime.now().isoformat()})
            return False
    
    def _initialize_progress(self, source_file: str, paths: Dict[str, Path], 
                        config: Dict[str, Any], sections: Optional[List[int]], 
                        skip_rvc: bool):
        """Initialize progress - NO CONFIG STORAGE"""
        self.progress = {
            'status': 'started',
            'start_time': datetime.now().isoformat(),
            'source_file': str(source_file),
            'skip_rvc': skip_rvc,
            'project_name': config['metadata']['project_name'],      # Just basics
            'batch_name': config['metadata']['batch_name'],
            'tts_engine': config['metadata']['tts_engine'],
            'rvc_voice': config['metadata']['rvc_voice'],
            'paths': {k: str(v) for k, v in paths.items()},
            'phases': {
                'preprocessing': {'complete': False},
                'section_processing': {'complete': False},
                'cleanup': {'complete': False}
            },
            'sections': {
                'total': 0,
                'completed': [],
                'current': None,
                'remaining': [],
                'files': {}
            }
        }
        self._save_progress()
    
    def _run_preprocessing(self) -> bool:
        """Phase 1: Text preprocessing and section creation"""
        print(f"\n📝 Phase 1: Preprocessing and Section Creation")
        
        start_time = time.time()
        
        try:
            # Import preprocessing modules
            from preprocessing.text_processor import preprocess_file
            from core.section_manager import SectionManager
            
            source_file = self.progress['source_file']
            config = self.progress['config']
            
            # Step 1: Extract and clean text
            clean_text_file = Path(self.progress['paths']['batch_dir']) / "clean_text.txt"
            if not preprocess_file(source_file, clean_text_file, config):
                print(f"❌ Text preprocessing failed")
                return False
            
            # Step 2: Split into sections
            with open(clean_text_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            section_manager = SectionManager(config)
            sections = section_manager.split_text_into_sections(text)
            
            # Step 3: Save sections
            batch_dir = Path(self.progress['paths']['batch_dir'])
            section_files = section_manager.save_sections(sections, batch_dir)
            
            # Update progress
            self.progress['sections']['total'] = len(sections)
            self.progress['sections']['remaining'] = list(range(1, len(sections) + 1))
            self.progress['sections']['section_files'] = [str(f) for f in section_files]
            
            self._mark_phase_complete('preprocessing')
            
            duration = time.time() - start_time
            self.progress['timing']['preprocessing'] = duration
            self._save_progress()
            
            print(f"✅ Preprocessing complete: {len(sections)} sections created")
            return True
            
        except Exception as e:
            print(f"❌ Preprocessing failed: {e}")
            return False
    
    def _run_section_processing(self) -> bool:
        """Phase 2: Process each section through TTS → RVC → Combine"""
        print(f"\n🎤 Phase 2: Section Processing")
        
        sections_info = self.progress['sections']
        total_sections = sections_info['total']
        remaining_sections = sections_info['remaining']
        
        if not remaining_sections:
            print(f"✅ All sections already processed")
            return True
        
        print(f"📊 Progress: {len(sections_info['completed'])}/{total_sections} sections complete")
        print(f"🔄 Remaining: {remaining_sections}")
        
        # Process each remaining section
        for section_num in remaining_sections[:]:
            if not self._process_single_section(section_num):
                return False
            
        self._mark_phase_complete('section_processing')
        return True
    
    def _process_single_section(self, section_num: int) -> bool:
        """Process a single section: TTS → RVC → Combine"""
        print(f"\n🎯 Processing Section {section_num}")
        
        self.progress['sections']['current'] = section_num
        self._save_progress()
        
        try:
            batch_dir = Path(self.progress['paths']['batch_dir'])
            section_file = batch_dir / "sections" / f"section_{section_num:03d}.txt"
            
            if not section_file.exists():
                print(f"❌ Section file not found: {section_file}")
                return False
            
            # Step 1: TTS Generation
            if not self._run_section_tts(section_num, section_file):
                return False
            
            # Step 2: RVC Processing (if not skipped)
            if not self.progress['skip_rvc']:
                if not self._run_section_rvc(section_num):
                    return False
            
            # Mark section as complete BEFORE master combination
            # Add safety checks to prevent list errors
            remaining_sections = self.progress['sections']['remaining']
            if section_num in remaining_sections:
                remaining_sections.remove(section_num)
            
            if section_num not in self.progress['sections']['completed']:
                self.progress['sections']['completed'].append(section_num)
            
            self.progress['sections']['current'] = None
            self._save_progress()
            
            print(f"✅ Section {section_num} complete ({len(self.progress['sections']['completed'])}/{self.progress['sections']['total']})")
            
            # Step 3: Add to master file (this can fail without losing section progress)
            if not self._combine_with_master(section_num):
                print(f"⚠️ Master combination failed, but section {section_num} processing is complete")
                # Don't return False - the section work is done, just master combination failed
            
            return True
            
        except Exception as e:
            print(f"❌ Section {section_num} failed: {e}")
            return False
            
    def _run_section_tts(self, section_num: int, section_file: Path) -> bool:
        """Run TTS engine on a single section - CLEANED UP"""
        print(f"  🎤 TTS Generation...")
        
        try:
            from engines import get_engine_processor
            
            # Load config from file (don't use stored config)
            config_path = Path(self.progress['paths']['job_config'])
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            engine_name = config['metadata']['tts_engine']
            
            # Use single temp folder for all temp work
            temp_dir = Path(self.progress['paths']['temp_dir'])
            temp_dir.mkdir(exist_ok=True)
            
            # Get engine processor and run
            processor = get_engine_processor(engine_name)
            generated_files = processor(str(section_file), str(temp_dir), config, self.progress['paths'])
            
            if not generated_files:
                print(f"❌ TTS generation failed for section {section_num}")
                return False
            
            # Combine TTS chunks for this section
            from audio.audio_combiner import combine_audio_files
            
            # Save TTS output to sections folder
            sections_dir = Path(self.progress['paths']['sections_dir'])
            section_tts_file = sections_dir / f"section_{section_num:03d}_tts.wav"
            silence_gap = config['audio']['silence_gap']
            
            if not combine_audio_files(generated_files, str(section_tts_file), silence_gap):
                print(f"❌ TTS audio combination failed for section {section_num}")
                return False
            
            # Store TTS file path
            self.progress['sections']['files'][f'section_{section_num}_tts'] = str(section_tts_file)
            self._save_progress()
            
            print(f"    ✅ TTS complete")
            return True
            
        except Exception as e:
            print(f"❌ TTS failed for section {section_num}: {e}")
            return False
    
    def _run_section_rvc(self, section_num: int) -> bool:
        """Run RVC processing on a single section"""
        print(f"  🎭 RVC Processing...")
        
        try:
            from audio.rvc_processor import process_audio_through_rvc
            
            batch_dir = Path(self.progress['paths']['batch_dir'])
            tts_file = self.progress['sections']['files'][f'section_{section_num}_tts']
            rvc_file = batch_dir / f"section_{section_num:03d}_rvc.wav"
            
            config = self.progress['config']
            
            if not process_audio_through_rvc(tts_file, str(rvc_file), config):
                print(f"❌ RVC processing failed for section {section_num}")
                return False
            
            # Store RVC file path
            self.progress['sections']['files'][f'section_{section_num}_rvc'] = str(rvc_file)
            self._save_progress()
            
            print(f"    ✅ RVC complete")
            return True
            
        except Exception as e:
            print(f"❌ RVC failed for section {section_num}: {e}")
            return False
    
    def _combine_with_master(self, section_num: int) -> bool:
        """Add completed section to master file"""
        print(f"  🔗 Adding to master file...")
        
        try:
            from audio.audio_combiner import combine_master_file
            
            batch_dir = Path(self.progress['paths']['batch_dir'])
            master_file = Path(self.progress['paths']['final'])
            
            # Determine which file to use (RVC or TTS)
            if self.progress['skip_rvc']:
                section_file = self.progress['sections']['files'][f'section_{section_num}_tts']
            else:
                section_file = self.progress['sections']['files'][f'section_{section_num}_rvc']
            
            if not combine_master_file(str(section_file), str(master_file)):
                print(f"❌ Failed to add section {section_num} to master file")
                return False
            
            print(f"    ✅ Added to master")
            return True
            
        except Exception as e:
            print(f"❌ Master combination failed for section {section_num}: {e}")
            return False
    
    def _run_cleanup(self):
        """Delete entire temp folder when done"""
        try:
            temp_dir = Path(self.progress['paths']['temp_dir'])
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)  # Delete entire temp folder
                print(f"  ✅ Removed temp directory")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")
    
    def _load_progress(self) -> Dict[str, Any]:
        """Load progress from file"""
        if not self.progress_file:
            return {}
        
        progress_path = Path(self.progress_file)
        if not progress_path.exists():
            return {}
        
        try:
            with open(progress_path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    
    def _save_progress(self):
        """Save progress to file"""
        if self.progress_file:
            progress_path = Path(self.progress_file)
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            with open(progress_path, 'w') as f:
                json.dump(self.progress, f, indent=2)
    
    def _update_progress(self, updates: Dict[str, Any]):
        """Update progress with new data"""
        self.progress.update(updates)
        self._save_progress()
    
    def _is_phase_complete(self, phase_name: str) -> bool:
        """Check if a phase is complete"""
        return self.progress.get('phases', {}).get(phase_name, {}).get('complete', False)
    
    def _mark_phase_complete(self, phase_name: str, data: Optional[Dict[str, Any]] = None):
        """Mark a phase as complete"""
        if 'phases' not in self.progress:
            self.progress['phases'] = {}
        
        self.progress['phases'][phase_name] = {
            'complete': True,
            'completed_at': datetime.now().isoformat(),
            'data': data or {}
        }
        self._save_progress()