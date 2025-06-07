#!/usr/bin/env python3
"""
Pipeline Manager - Simplified without progress bars
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Simplified imports - no complex progress handling
from core.progress_display_manager import log_error, log_info, log_success, log_warning

class PipelineManager:
    """Manages section-based audiobook pipeline with resume capability and simple text logging"""
    
    def __init__(self):
        self.progress = {}
        self.progress_file = None
        self.project_start_time = None
    
    def run(self, source_file: str, paths: Dict[str, Path], config: Dict[str, Any], 
            sections: Optional[List[int]] = None, skip_rvc: bool = False) -> bool:
        """Execute the section-based pipeline"""
        
        self.progress_file = paths['log']
        self.project_start_time = time.time()
        
        try:
            # Load or initialize progress (simplified - no config storage)
            self.progress = self._load_progress()
            
            # Initialize if starting fresh
            if 'status' not in self.progress:
                self._initialize_progress(source_file, paths, config, sections, skip_rvc)
            
            # Log pipeline start
            self._log_checkpoint("PIPELINE_START", "Started AudiobookGenerator pipeline")
            
            # Simple header output
            print("🚀 Section-based Pipeline", file=sys.stderr)
            project_name = config.get('metadata', {}).get('project_name', 'Unknown')
            tts_engine = config.get('metadata', {}).get('tts_engine', 'unknown').upper()
            rvc_voice = config.get('metadata', {}).get('rvc_voice', 'none')
            print(f"📁 Project: {project_name} | 🎤 Engine: {tts_engine} | 🎭 RVC: {rvc_voice}", file=sys.stderr)
            print()
            
            # Phase 1: Preprocessing and section creation
            if not self._is_phase_complete('preprocessing'):
                if not self._run_preprocessing():
                    return False
            
            # Phase 2: Section processing loop
            if not self._run_section_processing():
                return False
            
            # Phase 3: Cleanup
            self._run_cleanup()
            
            # Log pipeline completion
            pipeline_end_time = datetime.now().isoformat()
            if 'start_time' in self.progress:
                start_time = datetime.fromisoformat(self.progress['start_time'])
                end_time = datetime.now()
                total_duration = (end_time - start_time).total_seconds()
                self._log_checkpoint("PIPELINE_COMPLETE", f"Pipeline completed successfully in {total_duration:.1f}s total")
            else:
                self._log_checkpoint("PIPELINE_COMPLETE", "Pipeline completed successfully")
            
            print("\n🎉 Pipeline completed successfully!", file=sys.stderr)
            final_path = self.progress['paths']['final']
            print(f"🎵 Final audio: {final_path}", file=sys.stderr)
            
            return True
            
        except Exception as e:
            print(f"❌ Pipeline error: {e}", file=sys.stderr)
            self._log_checkpoint("PIPELINE_ERROR", f"Pipeline failed: {e}")
            self._update_progress({'error': str(e), 'failed_at': datetime.now().isoformat()})
            return False
    
    def _initialize_progress(self, source_file: str, paths: Dict[str, Path], 
                        config: Dict[str, Any], sections: Optional[List[int]], 
                        skip_rvc: bool):
        """Initialize progress - simplified"""
        self.progress = {
            'status': 'started',
            'start_time': datetime.now().isoformat(),
            'source_file': str(source_file),
            'skip_rvc': skip_rvc,
            'project_name': config['metadata']['project_name'],
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
        print("📝 Phase 1: Preprocessing and Section Creation", file=sys.stderr)
        
        start_time = time.time()
        self._log_checkpoint("PREPROCESSING_START", "Started text preprocessing and section creation")
        
        try:
            # Import preprocessing modules
            from preprocessing.text_processor import preprocess_file
            from core.section_manager import SectionManager
            
            source_file = self.progress['source_file']
            
            # Load config from job config file (not stored in progress)
            config_path = Path(self.progress['paths']['job_config'])
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Step 1: Extract and clean text
            batch_dir = Path(self.progress['paths']['batch_dir'])
            clean_text_file = batch_dir / "clean_text.txt"
            
            text_start_time = time.time()
            if not preprocess_file(source_file, clean_text_file, config):
                print("❌ Text preprocessing failed", file=sys.stderr)
                self._log_checkpoint("TEXT_PROCESSING_FAILED", "Text preprocessing failed")
                return False
            text_duration = time.time() - text_start_time
            self._log_checkpoint("TEXT_PROCESSING_COMPLETE", f"Text preprocessing completed in {text_duration:.1f}s")
            
            # Step 2: Split into sections
            with open(clean_text_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            section_manager = SectionManager(config)
            sections = section_manager.split_text_into_sections(text)
            
            # Step 3: Save sections to organized sections folder
            sections_dir = Path(self.progress['paths']['sections_dir'])
            section_files = section_manager.save_sections(sections, batch_dir)
            
            # Update progress
            self.progress['sections']['total'] = len(sections)
            self.progress['sections']['remaining'] = list(range(1, len(sections) + 1))
            self.progress['sections']['section_files'] = [str(f) for f in section_files]
            
            self._mark_phase_complete('preprocessing')
            
            duration = time.time() - start_time
            self._log_checkpoint("PREPROCESSING_COMPLETE", f"Preprocessing completed in {duration:.1f}s - created {len(sections)} sections")
            print(f"✅ Preprocessing complete: {len(sections)} sections created", file=sys.stderr)
            return True
            
        except Exception as e:
            print(f"❌ Preprocessing failed: {e}", file=sys.stderr)
            self._log_checkpoint("PREPROCESSING_ERROR", f"Preprocessing failed: {e}")
            return False
    
    def _run_section_processing(self) -> bool:
        """Phase 2: Process each section through TTS → RVC → Combine"""
        print("\n🎤 Phase 2: Section Processing", file=sys.stderr)
        
        sections_info = self.progress['sections']
        total_sections = sections_info['total']
        remaining_sections = sections_info['remaining']
        
        if not remaining_sections:
            print("✅ All sections already processed", file=sys.stderr)
            return True
        
        # Process each remaining section
        for section_num in remaining_sections[:]:
            if not self._process_single_section(section_num):
                return False
            
        self._mark_phase_complete('section_processing')
        return True
    
    def _process_single_section(self, section_num: int) -> bool:
        """Process a single section: TTS → RVC → Combine"""
        section_start_time = time.time()
        
        print(f"\n🎯 Processing Section {section_num}", file=sys.stderr)
        
        self.progress['sections']['current'] = section_num
        self._save_progress()
        
        # Log section start
        self._log_checkpoint(f"SECTION_{section_num}_START", f"Started processing section {section_num}")
        
        try:
            sections_dir = Path(self.progress['paths']['sections_dir'])
            section_file = sections_dir / f"section_{section_num:03d}.txt"
            
            if not section_file.exists():
                print(f"❌ Section file not found: {section_file}", file=sys.stderr)
                self._log_checkpoint(f"SECTION_{section_num}_ERROR", f"Section file not found: {section_file}")
                return False
            
            # Step 1: TTS Generation
            tts_start_time = time.time()
            if not self._run_section_tts(section_num, section_file):
                self._log_checkpoint(f"SECTION_{section_num}_TTS_FAILED", "TTS generation failed")
                return False
            tts_duration = time.time() - tts_start_time
            self._log_checkpoint(f"SECTION_{section_num}_TTS_COMPLETE", f"TTS completed in {tts_duration:.1f}s")
            
            # Step 2: RVC Processing (if not skipped)
            if not self.progress['skip_rvc']:
                rvc_start_time = time.time()
                if not self._run_section_rvc(section_num):
                    self._log_checkpoint(f"SECTION_{section_num}_RVC_FAILED", "RVC processing failed")
                    return False
                rvc_duration = time.time() - rvc_start_time
                self._log_checkpoint(f"SECTION_{section_num}_RVC_COMPLETE", f"RVC completed in {rvc_duration:.1f}s")
            else:
                self._log_checkpoint(f"SECTION_{section_num}_RVC_SKIPPED", "RVC processing skipped")
            
            # Mark section as complete BEFORE master combination
            remaining_sections = self.progress['sections']['remaining']
            if section_num in remaining_sections:
                remaining_sections.remove(section_num)
            
            if section_num not in self.progress['sections']['completed']:
                self.progress['sections']['completed'].append(section_num)
            
            self.progress['sections']['current'] = None
            self._save_progress()
            
            # Simple completion message
            completed_count = len(self.progress['sections']['completed'])
            total_sections = self.progress['sections']['total']
            print(f"✅ Section {section_num} complete ({completed_count}/{total_sections})", file=sys.stderr)
            
            # Step 3: Add to master file (this can fail without losing section progress)
            master_start_time = time.time()
            if not self._combine_with_master(section_num):
                print(f"⚠️ Master combination failed, but section {section_num} processing is complete", file=sys.stderr)
                self._log_checkpoint(f"SECTION_{section_num}_MASTER_FAILED", "Master combination failed, but section processing complete")
                # Don't return False - the section work is done, just master combination failed
            else:
                master_duration = time.time() - master_start_time
                self._log_checkpoint(f"SECTION_{section_num}_MASTER_COMPLETE", f"Added to master in {master_duration:.1f}s")
            
            # Log section completion
            section_duration = time.time() - section_start_time
            self._log_checkpoint(f"SECTION_{section_num}_COMPLETE", f"Section {section_num} completed in {section_duration:.1f}s total")
            
            return True
            
        except Exception as e:
            print(f"❌ Section {section_num} failed: {e}", file=sys.stderr)
            self._log_checkpoint(f"SECTION_{section_num}_ERROR", f"Section {section_num} failed: {e}")
            return False
            
    def _run_section_tts(self, section_num: int, section_file: Path) -> bool:
        """Run TTS engine on a single section"""
        print("  🎤 TTS Generation...", file=sys.stderr)
        
        try:
            from engines import get_engine_processor
            
            # Load config from job config file
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
                print(f"❌ TTS generation failed for section {section_num}", file=sys.stderr)
                return False
            
            # Combine TTS chunks for this section
            from audio.audio_combiner import combine_audio_files
            
            # Save TTS output to sections folder
            sections_dir = Path(self.progress['paths']['sections_dir'])
            section_tts_file = sections_dir / f"section_{section_num:03d}_tts.wav"
            silence_gap = config['audio']['silence_gap']
            
            if not combine_audio_files(generated_files, str(section_tts_file), silence_gap):
                print(f"❌ TTS audio combination failed for section {section_num}", file=sys.stderr)
                return False
            
            # Store TTS file path
            self.progress['sections']['files'][f'section_{section_num}_tts'] = str(section_tts_file)
            self._save_progress()
            
            print("✅ TTS complete", file=sys.stderr)
            return True
            
        except Exception as e:
            print(f"❌ TTS failed for section {section_num}: {e}", file=sys.stderr)
            return False
    
    def _run_section_rvc(self, section_num: int) -> bool:
        print("  🎭 RVC Processing...", file=sys.stderr)
        
        try:
            from audio.rvc_processor import process_audio_through_rvc
            
            # Load config from job config file
            config_path = Path(self.progress['paths']['job_config'])
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Get section files from sections folder
            sections_dir = Path(self.progress['paths']['sections_dir'])
            tts_file = self.progress['sections']['files'][f'section_{section_num}_tts']
            rvc_file = sections_dir / f"section_{section_num:03d}_rvc.wav"
            
            if not process_audio_through_rvc(tts_file, str(rvc_file), config):
                print(f"❌ RVC processing failed for section {section_num}", file=sys.stderr)
                return False
            
            # Store RVC file path
            self.progress['sections']['files'][f'section_{section_num}_rvc'] = str(rvc_file)
            self._save_progress()
            
            print("✅ RVC complete", file=sys.stderr)
            return True
            
        except Exception as e:
            print(f"❌ RVC failed for section {section_num}: {e}", file=sys.stderr)
            return False
    
    def _combine_with_master(self, section_num: int) -> bool:
        """Add completed section to master file"""
        
        try:
            from audio.audio_combiner import combine_master_file
            
            master_file = Path(self.progress['paths']['final'])
            
            # Determine which file to use (RVC or TTS)
            if self.progress['skip_rvc']:
                section_file = self.progress['sections']['files'][f'section_{section_num}_tts']
            else:
                section_file = self.progress['sections']['files'][f'section_{section_num}_rvc']
            
            if not combine_master_file(str(section_file), str(master_file)):
                print(f"❌ Failed to add section {section_num} to master file", file=sys.stderr)
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Master combination failed for section {section_num}: {e}", file=sys.stderr)
            return False
    
    def _run_cleanup(self):
        """Delete entire temp folder when done"""
        cleanup_start_time = time.time()
        self._log_checkpoint("CLEANUP_START", "Started cleanup process")
        
        try:
            temp_dir = Path(self.progress['paths']['temp_dir'])
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)  # Delete entire temp folder
                self._log_checkpoint("TEMP_CLEANUP_COMPLETE", "Removed temporary directory")
                
            # Mark cleanup complete
            self._mark_phase_complete('cleanup')
            cleanup_duration = time.time() - cleanup_start_time
            self._log_checkpoint("CLEANUP_COMPLETE", f"Cleanup completed in {cleanup_duration:.1f}s")
            
        except Exception as e:
            # Log warning to progress file instead of console
            self._log_checkpoint("CLEANUP_WARNING", f"Cleanup warning: {e}")
    
    def _log_checkpoint(self, checkpoint_type: str, message: str):
        """Log a checkpoint with timestamp to progress file"""
        timestamp = datetime.now().isoformat()
        
        # Initialize checkpoints list if not exists
        if 'checkpoints' not in self.progress:
            self.progress['checkpoints'] = []
        
        checkpoint = {
            'timestamp': timestamp,
            'type': checkpoint_type,
            'message': message
        }
        
        self.progress['checkpoints'].append(checkpoint)
        self._save_progress()
        
        # Simple stderr logging
        print(f"CHECKPOINT [{timestamp}] {checkpoint_type}: {message}", file=sys.stderr)
    
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