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
        
        # Log more descriptive initial checkpoint
        project_name = config['metadata']['project_name']
        tts_engine = config['metadata']['tts_engine'].upper()
        rvc_voice = config['metadata']['rvc_voice']
        skip_rvc_text = " (RVC disabled)" if skip_rvc else f" + {rvc_voice}"
        
        self._log_checkpoint("PIPELINE_START", f"Started {project_name} using {tts_engine}{skip_rvc_text}")
    
    def _run_preprocessing(self) -> bool:
        """Phase 1: Text preprocessing and section creation"""
        print("📝 Phase 1: Preprocessing and Section Creation", file=sys.stderr)
        
        start_time = time.time()
        self._log_checkpoint("PREPROCESSING_START", "Beginning text extraction and section creation")
        
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
                self._log_checkpoint("TEXT_PROCESSING_FAILED", "Failed to extract and clean text from source")
                return False
            text_duration = time.time() - text_start_time
            
            # Read text to get word count for logging
            with open(clean_text_file, 'r', encoding='utf-8') as f:
                text = f.read()
            word_count = len(text.split())
            
            self._log_checkpoint("TEXT_PROCESSING_COMPLETE", f"Extracted {word_count:,} words in {text_duration:.1f}s")
            
            # Step 2: Split into sections
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
            self._log_checkpoint("PREPROCESSING_COMPLETE", f"Created {len(sections)} sections (~30min each) in {duration:.1f}s")
            print(f"✅ Preprocessing complete: {len(sections)} sections created", file=sys.stderr)
            return True
            
        except Exception as e:
            print(f"❌ Preprocessing failed: {e}", file=sys.stderr)
            self._log_checkpoint("PREPROCESSING_ERROR", f"Preprocessing failed: {str(e)}")
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
            
            try:
                sections_dir = Path(self.progress['paths']['sections_dir'])
                section_file = sections_dir / f"section_{section_num:03d}.txt"
                
                if not section_file.exists():
                    print(f"❌ Section file not found: {section_file}", file=sys.stderr)
                    self._log_checkpoint(f"SECTION_{section_num}_ERROR", f"Section {section_num} file not found")
                    return False
                
                # Log section start
                with open(section_file, 'r', encoding='utf-8') as f:
                    section_text = f.read()
                word_count = len(section_text.split())
                estimated_audio_minutes = word_count / 150  # 150 words per minute
                
                self._log_checkpoint(f"SECTION_{section_num}_START", f"Section {section_num}: {word_count} words (~{estimated_audio_minutes:.1f}min audio)")
                
                # Step 1: TTS Generation
                tts_start_time = time.time()
                if not self._run_section_tts(section_num, section_file):
                    self._log_checkpoint(f"SECTION_{section_num}_TTS_FAILED", f"Section {section_num} TTS generation failed")
                    return False
                tts_duration = time.time() - tts_start_time
                self._log_checkpoint(f"SECTION_{section_num}_TTS_COMPLETE", f"Section {section_num} TTS: {word_count} words → audio in {tts_duration:.1f}s")
                
                # Step 2: RVC Processing (if not skipped)
                if not self.progress['skip_rvc']:
                    rvc_start_time = time.time()
                    if not self._run_section_rvc(section_num):
                        self._log_checkpoint(f"SECTION_{section_num}_RVC_FAILED", f"Section {section_num} RVC processing failed")
                        return False
                    rvc_duration = time.time() - rvc_start_time
                    self._log_checkpoint(f"SECTION_{section_num}_RVC_COMPLETE", f"Section {section_num} RVC: voice conversion in {rvc_duration:.1f}s")
                else:
                    self._log_checkpoint(f"SECTION_{section_num}_RVC_SKIPPED", f"Section {section_num} RVC processing skipped")
                
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
                    self._log_checkpoint(f"SECTION_{section_num}_MASTER_FAILED", f"Section {section_num} master combination failed (section saved)")
                    # Don't return False - the section work is done, just master combination failed
                else:
                    master_duration = time.time() - master_start_time
                    self._log_checkpoint(f"SECTION_{section_num}_MASTER_COMPLETE", f"Section {section_num} added to master file in {master_duration:.1f}s")
                
                # Log section completion
                section_duration = time.time() - section_start_time
                self._log_checkpoint(f"SECTION_{section_num}_COMPLETE", f"Section {section_num} fully completed: {word_count} words in {section_duration:.1f}s total")
                
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
            
            # Track RVC timing for ETA
            rvc_start_time = time.time()
            
            # Show RVC progress with ETA
            total_sections = self.progress['sections']['total']
            completed_sections = len(self.progress['sections']['completed'])
            self._show_rvc_progress(section_num, total_sections, completed_sections)
            
            if not process_audio_through_rvc(tts_file, str(rvc_file), config):
                print(f"❌ RVC processing failed for section {section_num}", file=sys.stderr)
                return False
            
            # Record RVC timing for future ETA calculations
            rvc_duration = time.time() - rvc_start_time
            if 'rvc_times' not in self.progress:
                self.progress['rvc_times'] = []
            self.progress['rvc_times'].append(rvc_duration)
            self._save_progress()
            
            # Store RVC file path
            self.progress['sections']['files'][f'section_{section_num}_rvc'] = str(rvc_file)
            self._save_progress()
            
            print("✅ RVC complete", file=sys.stderr)
            return True
            
        except Exception as e:
            print(f"❌ RVC failed for section {section_num}: {e}", file=sys.stderr)
            return False
    
    def _show_rvc_progress(self, current_section: int, total_sections: int, completed_sections: int):
        """Show RVC progress bar with ETA based on previous RVC timings"""
        # Calculate which section we're processing in RVC terms
        rvc_completed = completed_sections  # Sections that have completed RVC
        rvc_remaining = total_sections - completed_sections  # Including current section
        
        # Calculate ETA based on previous RVC times
        if hasattr(self, 'progress') and 'rvc_times' in self.progress and self.progress['rvc_times']:
            rvc_times = self.progress['rvc_times']
            if len(rvc_times) >= 2:
                # Use average of last few RVC times
                recent_times = rvc_times[-3:] if len(rvc_times) >= 3 else rvc_times
                avg_rvc_time = sum(recent_times) / len(recent_times)
            else:
                avg_rvc_time = rvc_times[0]
            
            # Estimate remaining time
            remaining_time = (rvc_remaining - 1) * avg_rvc_time  # -1 because current section is in progress
            
            if remaining_time < 60:
                eta_str = f"{int(remaining_time)}s"
            elif remaining_time < 3600:
                minutes = int(remaining_time // 60)
                seconds = int(remaining_time % 60)
                eta_str = f"{minutes}m {seconds}s"
            else:
                hours = int(remaining_time // 3600)
                minutes = int((remaining_time % 3600) // 60)
                eta_str = f"{hours}h {minutes}m"
        else:
            eta_str = "calculating..."
        
        # Create RVC progress bar
        percent = (rvc_completed / total_sections) * 100 if total_sections > 0 else 0
        
        # Get terminal width
        try:
            import shutil
            terminal_width = shutil.get_terminal_size().columns
        except:
            terminal_width = 80
        
        # Build progress bar components
        prefix = "    🎭 RVC: "
        section_info = f"{rvc_completed}/{total_sections} sections"
        percent_info = f"({percent:.0f}%)"
        eta_info = f"ETA: {eta_str}"
        
        # Build suffix with proper spacing
        suffix = f" {section_info} {percent_info} {eta_info}"
        
        # Calculate available space for the bar
        total_text_length = len(prefix) + len(suffix) + 2  # +2 for brackets []
        available_width = terminal_width - total_text_length - 5  # -5 for extra safety
        bar_width = max(5, min(30, available_width))  # Conservative bar width
        
        # Create progress bar
        if total_sections > 0:
            filled_length = int(bar_width * rvc_completed // total_sections)
        else:
            filled_length = bar_width
        bar = '█' * filled_length + '░' * (bar_width - filled_length)
        
        # Build complete line
        progress_line = f"{prefix}[{bar}]{suffix}"
        
        # Final safety check - if still too long, truncate the bar more
        while len(progress_line) > terminal_width - 2 and bar_width > 5:
            bar_width -= 1
            if total_sections > 0:
                filled_length = int(bar_width * rvc_completed // total_sections)
            else:
                filled_length = bar_width
            bar = '█' * filled_length + '░' * (bar_width - filled_length)
            progress_line = f"{prefix}[{bar}]{suffix}"
        
        # Print the progress bar
        print(f"{progress_line}", file=sys.stderr)
    
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
        """Load progress from file and reconstruct state from checkpoints"""
        if not self.progress_file:
            return {}
        
        progress_path = Path(self.progress_file)
        if not progress_path.exists():
            return {}
        
        try:
            with open(progress_path, 'r') as f:
                progress = json.load(f)
            
            # Reconstruct section state from checkpoints
            self._reconstruct_section_state_from_checkpoints(progress)
            
            return progress
        except Exception:
            return {}
    
    def _reconstruct_section_state_from_checkpoints(self, progress: Dict[str, Any]):
        """Reconstruct section state from checkpoints - allows simple checkpoint deletion"""
        checkpoints = progress.get('checkpoints', [])
        sections = progress.get('sections', {})
        
        # Get total sections (this should be preserved)
        total_sections = sections.get('total', 0)
        if total_sections == 0:
            return  # Can't reconstruct without knowing total
        
        # Initialize tracking
        completed_sections = set()
        section_files = {}
        current_section = None
        
        # Process checkpoints to determine current state
        for checkpoint in checkpoints:
            checkpoint_type = checkpoint.get('type', '')
            
            if 'SECTION_' in checkpoint_type and '_' in checkpoint_type:
                try:
                    # Extract section number from checkpoint type like "SECTION_4_TTS_COMPLETE"
                    parts = checkpoint_type.split('_')
                    section_num = int(parts[1])
                    action = '_'.join(parts[2:])  # e.g., "TTS_COMPLETE", "RVC_COMPLETE", "COMPLETE"
                    
                    # Track current section being processed
                    if action in ['START', 'TTS_COMPLETE', 'RVC_COMPLETE', 'MASTER_COMPLETE', 'MASTER_FAILED']:
                        current_section = section_num
                    
                    # Track completed files
                    if action == 'TTS_COMPLETE':
                        # TTS file exists
                        section_files[f'section_{section_num}_tts'] = sections.get('files', {}).get(f'section_{section_num}_tts', f"sections/section_{section_num:03d}_tts.wav")
                    
                    elif action == 'RVC_COMPLETE':
                        # RVC file exists (TTS should already be tracked)
                        section_files[f'section_{section_num}_rvc'] = sections.get('files', {}).get(f'section_{section_num}_rvc', f"sections/section_{section_num:03d}_rvc.wav")
                    
                    elif action == 'COMPLETE':
                        # Section fully completed
                        completed_sections.add(section_num)
                        current_section = None  # No longer current
                
                except (ValueError, IndexError):
                    continue  # Skip malformed checkpoint types
        
        # Determine remaining sections
        all_sections = set(range(1, total_sections + 1))
        remaining_sections = all_sections - completed_sections
        
        # If we have a current section, make sure it's in remaining
        if current_section:
            remaining_sections.add(current_section)
        
        # Update progress with reconstructed state
        sections.update({
            'completed': sorted(list(completed_sections)),
            'remaining': sorted(list(remaining_sections)),
            'current': current_section,
            'files': section_files
        })
        
        progress['sections'] = sections
        
        # Debug info
        print(f"🔄 Reconstructed state from {len(checkpoints)} checkpoints:", file=sys.stderr)
        print(f"   Completed sections: {sorted(list(completed_sections))}", file=sys.stderr)
        print(f"   Current section: {current_section}", file=sys.stderr)
        print(f"   Remaining sections: {sorted(list(remaining_sections))}", file=sys.stderr)
        print(f"   Tracked files: {len(section_files)}", file=sys.stderr)
    
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