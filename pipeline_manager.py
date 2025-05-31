#!/usr/bin/env python3
"""
Pipeline Manager - Orchestrates the 5-phase audiobook generation pipeline
Updated to work with the new modular engine architecture
"""

import os
import sys
import json
import shutil
import time
from pathlib import Path
from datetime import datetime

class PipelineManager:
    """Manages pipeline execution and progress tracking"""

    def __init__(self):
        self.progress = {}
        self.log_file = None

    def run(self, source_file, paths, config, sections=None, skip_rvc=False):
        """Execute the complete pipeline"""
        self.log_file = paths['log']
        self.progress = self._load_progress()

        print(f"📊 Progress log: {self.log_file}")

        try:
            total_start = time.time()

            if not self._time_and_run("preprocessing", self._run_preprocessing, source_file, paths, config, sections):
                return False

            generated_files = self._time_and_run("tts_generation", self._run_tts_generation, paths, config)
            if not generated_files:
                return False

            if not self._time_and_run("audio_combination", self._run_audio_combination, generated_files, paths, config):
                return False

            if not self._time_and_run("rvc_conversion", self._run_rvc_conversion, paths, config, skip_rvc):
                return False

            self._time_and_run("cleanup", self._run_cleanup, paths, config, skip_rvc)
            self._mark_phase_complete('pipeline', {'final': str(paths['final'])})

            total_end = time.time()
            total_time = round(total_end - total_start, 2)
            self.progress['timing']['total_time'] = total_time
            self._save_progress()

            print("\n🚀 Pipeline Summary")
            for phase, seconds in self.progress['timing'].items():
                ms = int((seconds - int(seconds)) * 100)
                h = int(seconds) // 3600
                m = (int(seconds) % 3600) // 60
                s = int(seconds) % 60
                formatted = f"{h:02}:{m:02}:{s:02}.{ms:02}"
                print(f"  {phase:20} - {formatted}")

            return True

        except Exception as e:
            print(f"❌ Pipeline error: {e}")
            return False

    def _time_and_run(self, label, func, *args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        duration = round(end - start, 2)

        ms = int((duration - int(duration)) * 100)
        h = int(duration) // 3600
        m = (int(duration) % 3600) // 60
        s = int(duration) % 60
        formatted = f"{h:02}:{m:02}:{s:02}.{ms:02}"
        print(f"⏱️ {label} completed in {formatted}")
        if 'timing' not in self.progress:
            self.progress['timing'] = {}
        self.progress['timing'][label] = duration
        self._save_progress()

        return result

    def _load_progress(self):
        if not self.log_file or not Path(self.log_file).exists():
            return {
                'status': 'started',
                'start_time': datetime.now().isoformat(),
                'phases': {}
            }

        try:
            with open(self.log_file, 'r') as f:
                return json.load(f)
        except:
            return {'status': 'started', 'phases': {}}

    def _save_progress(self):
        if self.log_file:
            Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, 'w') as f:
                json.dump(self.progress, f, indent=2)

    def _is_phase_complete(self, phase_name):
        return self.progress.get('phases', {}).get(phase_name, {}).get('complete', False)

    def _mark_phase_complete(self, phase_name, data=None):
        if 'phases' not in self.progress:
            self.progress['phases'] = {}

        self.progress['phases'][phase_name] = {
            'complete': True,
            'completed_at': datetime.now().isoformat(),
            'data': data or {}
        }
        self._save_progress()

    def _run_preprocessing(self, source_file, paths, config, sections):
        if self._is_phase_complete('preprocessing'):
            print("✅ Preprocessing already complete")
            return True

        print(f"\n📝 Phase 1: Preprocessing")

        try:
            from preprocessing import preprocess_file
            clean_text_file = paths['batch_dir'] / f"{paths['batch_dir'].name}_clean.txt"
            success = preprocess_file(source_file, clean_text_file, config, sections)

            if success:
                self._mark_phase_complete('preprocessing', {
                    'clean_text': str(clean_text_file)
                })
                return True
            return False

        except Exception as e:
            print(f"❌ Preprocessing failed: {e}")
            return False

    def _run_tts_generation(self, paths, config):
        if self._is_phase_complete('tts_generation'):
            print("✅ TTS Generation already complete")
            files = self.progress['phases']['tts_generation']['data'].get('files', [])
            return [Path(f) for f in files]

        print(f"\n🎤 Phase 2: TTS Generation")

        try:
            from engine_registry import process_with_engine, ensure_engine_config
            from audio_processor import ensure_audio_config
            tts_engine = config['metadata']['tts_engine']

            engine_config_updated = ensure_engine_config(config, tts_engine)
            audio_config_updated = ensure_audio_config(config)

            if engine_config_updated or audio_config_updated:
                config_path = paths['config']
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"STATUS: Updated config with engine defaults", file=sys.stderr)

            clean_text_file = paths['batch_dir'] / f"{paths['batch_dir'].name}_clean.txt"
            if not clean_text_file.exists():
                print(f"❌ Clean text file not found: {clean_text_file}")
                return None

            generated_files = process_with_engine(tts_engine, clean_text_file, paths['temp_dir'], config, paths)

            if generated_files:
                self._mark_phase_complete('tts_generation', {
                    'files': [str(f) for f in generated_files]
                })
                return [Path(f) for f in generated_files]

            return None

        except Exception as e:
            print(f"❌ TTS Generation failed: {e}")
            return None

    def _run_audio_combination(self, generated_files, paths, config):
        if self._is_phase_complete('audio_combination'):
            print("✅ Audio Combination already complete")
            return True

        print(f"\n🔗 Phase 3: Audio Combination")

        try:
            from audio_processor import combine_audio_files
            silence_gap = config['audio']['silence_gap']

            success = combine_audio_files([str(f) for f in generated_files], paths['combined'], silence_gap)

            if success:
                self._mark_phase_complete('audio_combination', {
                    'combined': str(paths['combined'])
                })
                return True

            return False

        except Exception as e:
            print(f"❌ Audio Combination failed: {e}")
            return False

    def _run_rvc_conversion(self, paths, config, skip_rvc):
        if self._is_phase_complete('rvc_conversion') or skip_rvc:
            print("✅ RVC Conversion skipped or already complete")
            return True

        print(f"\n🎧 Phase 4: RVC Conversion")

        try:
            from audio_processor import process_audio_through_rvc
            rvc_output_path = paths['final']
            rvc_success = process_audio_through_rvc(paths['combined'], rvc_output_path, config)

            if rvc_success:
                self._mark_phase_complete('rvc_conversion', {
                    'converted': str(rvc_output_path)
                })
                return True

            return False

        except Exception as e:
            print(f"❌ RVC Conversion failed: {e}")
            return False

    def _run_cleanup(self, paths, config, skip_rvc):
        """Phase 5: Cleanup"""
        print("\n🧹 Phase 5: Cleanup")

        pipeline_config = config['pipeline']

        try:
            if pipeline_config.get('cleanup_temp_files', True) and paths['temp_dir'].exists():
                shutil.rmtree(paths['temp_dir'])
                print("  ✅ Removed temp files")

            if (pipeline_config.get('cleanup_intermediate_files', True) and
                paths['combined'].exists() and not skip_rvc):
                paths['combined'].unlink()
                print("  ✅ Removed intermediate combined file")

            clean_text_file = paths['batch_dir'] / f"{paths['batch_dir'].name}_clean.txt"
            if clean_text_file.exists() and pipeline_config.get('cleanup_temp_files', True):
                clean_text_file.unlink()
                print("  ✅ Removed clean text file")

        except Exception as e:
            print(f"  ⚠️ Cleanup warning: {e}")
