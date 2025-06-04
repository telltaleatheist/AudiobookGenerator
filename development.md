# AudiobookGenerator Development Documentation v2.0

## Overview

AudiobookGenerator is a sophisticated Python-based audiobook generation system that converts text documents (EPUB, PDF, TXT) into high-quality audiobooks using multiple TTS engines and voice conversion technology. The system features a **dynamic configuration architecture** where all settings are externally controlled via JSON configuration files, eliminating hardcoded parameters.

## System Architecture

### Core Components

1. **Main Entry Point**: `AudiobookGenerator.py` - Command-line interface and orchestration
2. **Project Management**: `project_manager.py` - Handles project structure, configuration copying, and file management
3. **Pipeline Management**: `pipeline_manager.py` - Orchestrates the 5-phase processing pipeline with config snapshots
4. **Dynamic Engine Registry**: `engine_registry.py` - Plugin system with dynamic parameter loading utilities
5. **Text Preprocessing**: `preprocessing.py` + `preprocessing_pdf.py` - Text extraction and cleaning
6. **TTS Engines**: Individual engine implementations (bark_engine.py, edge_engine.py, f5_engine.py, xtts_engine.py)
7. **Audio Processing**: `audio_processor.py` - Audio combination, RVC conversion, and post-processing

### Dynamic Configuration Architecture

**Key Innovation**: All TTS engine parameters are **dynamically loaded from JSON configuration files**. No engine parameters are hardcoded in Python code.

#### Configuration Flow:
1. **Default Template**: `default_config.json` - Master template with all possible parameters
2. **Project Config**: Copied from default template during project creation
3. **Job Config Snapshot**: Complete configuration snapshot created for each processing job
4. **Dynamic Loading**: Engines automatically detect and use any parameters present in config

### Processing Pipeline (5 Phases)

1. **Config Snapshot Creation**: Complete configuration captured before processing begins
2. **Preprocessing**: Extract and clean text from source documents  
3. **TTS Generation**: Convert text to speech using selected TTS engine with dynamic parameters
4. **Audio Combination**: Combine individual audio chunks with silence gaps
5. **RVC Conversion**: Apply voice conversion using selected RVC model
6. **Cleanup**: Remove temporary files and finalize output

## Project Structure

```
output/
├── project_name/
│   ├── source/           # Input files (.epub, .pdf, .txt)
│   ├── samples/          # Voice reference audio for cloning
│   ├── config/           # Project configuration
│   │   └── config.json   # Project config (copied from default_config.json)
│   ├── jobs/             # Processing batches
│   │   └── batch_name/
│   │       ├── config.json         # COMPLETE config snapshot
│   │       ├── config_summary.txt  # Human-readable summary
│   │       ├── progress.log        # Processing progress
│   │       ├── temp_files/         # Temporary processing files
│   │       ├── batch_name_tts.wav  # Combined TTS audio
│   │       └── batch_name_rvc.wav  # Final RVC processed audio
│   └── README.md
├── default_config.json   # Master configuration template
```

## Dynamic Configuration System

### External Configuration Control

All engine parameters are defined in `default_config.json` and automatically detected by engines:

```json
{
  "f5": {
    "speed": 0.9,
    "nfe_step": 128,
    "cfg_strength": 1.5,
    "any_new_parameter": "automatically_detected"
  },
  "xtts": {
    "temperature": 0.75,
    "emotion_strength": 1.2,
    "future_parameter": "works_immediately"
  }
}
```

### Dynamic Parameter Loading

Engines use the **Enhanced Engine Registry** utilities:

```python
# Extract ALL parameters from config automatically
engine_config = extract_engine_config(config, 'f5', verbose=True)

# Create generation parameters with automatic filtering
generation_params = create_generation_params(
    base_params, 
    engine_config, 
    filter_function=api_function,  # Only passes valid parameters
    verbose=True
)
```

### Config Inheritance and Snapshots

1. **Project Creation**: `default_config.json` → `project/config/config.json`
2. **Job Execution**: Project config → `job/config.json` (complete snapshot)
3. **Result Analysis**: Compare job snapshots to identify optimal settings

## TTS Engine System

### Dynamic Engine Architecture

All engines use the same pattern:
- **No hardcoded parameters** - Everything from JSON
- **Automatic parameter detection** - Add any parameter to config, engine uses it
- **Parameter filtering** - Only valid parameters passed to TTS APIs
- **Enhanced logging** - Shows which parameters are being used

### Supported TTS Engines

#### 1. Bark (`bark_engine.py`)
- **Type**: Local neural TTS with voice presets
- **Dynamic Features**: All generation parameters (semantic_temp, coarse_temp, etc.)
- **Quality Control**: Artifact detection, voice consistency modes, advanced chunking
- **Memory Management**: Configurable model reloading, garbage collection

#### 2. EdgeTTS (`edge_engine.py`) 
- **Type**: Free Microsoft cloud TTS service
- **Scope**: FREE version only (rate, pitch, volume parameters)
- **Features**: Async processing, retry logic, voice availability checking
- **Limitations**: Warns about paid Azure features if present in config

#### 3. F5-TTS (`f5_engine.py`)
- **Type**: Advanced voice cloning with reference audio
- **Features**: Single-pass processing, companion text auto-detection
- **Dynamic Parameters**: All F5-TTS API parameters automatically supported

#### 4. XTTS (`xtts_engine.py`)
- **Type**: Coqui's multilingual voice cloning
- **Features**: Multiple reference samples, advanced prosody control
- **Dynamic Parameters**: Comprehensive parameter support for quality tuning

### Engine Registration and Loading

```python
# engines automatically register with no hardcoded config
def register_f5_engine():
    register_engine(
        name='f5',
        processor_func=process_f5_text_file
        # NO default_config parameter needed!
    )
```

## Enhanced Engine Registry

### Dynamic Parameter Utilities

**Core Functions:**
- `extract_engine_config()` - Gets all parameters from JSON automatically
- `filter_params_for_function()` - Only passes parameters the API accepts
- `create_generation_params()` - Merges and filters parameters
- `show_engine_config_summary()` - Displays active configuration

**Benefits:**
- ✅ Add ANY parameter to JSON → Engine automatically uses it
- ✅ Invalid parameters automatically filtered out
- ✅ No coding required for new TTS features
- ✅ Comprehensive logging shows exactly what's used

## Voice Cloning and Reference Audio

### Automatic Sample Detection

The system automatically detects voice samples:

```
project/samples/
├── my_voice.wav           # Audio file
├── my_voice.txt          # Optional: transcript for F5-TTS (auto-detected)
├── speaker2.wav          # Multiple samples supported by XTTS
└── speaker2.txt
```

**Detection Logic:**
- **F5-TTS**: Uses first `.wav` + matching `.txt` file (auto-transcribe if no text)
- **XTTS**: Uses all `.wav` files (supports multiple references)
- **Bark/Edge**: Use built-in voices (ignore samples directory)

## RVC (Real-time Voice Conversion)

### Multi-Voice RVC System

Supports multiple voice profiles with global + voice-specific settings:

```json
{
  "rvc_global": {
    "speed_factor": 1.0,
    "f0_method": "crepe",
    "clean_voice": true
  },
  "rvc_my_voice": {
    "model": "my_voice",
    "n_semitones": -2,
    "index_rate": 0.35
  },
  "rvc_sigma_male_narrator": {
    "model": "Sigma Male Narrator", 
    "n_semitones": -6,
    "index_rate": 0.3
  }
}
```

**Voice Selection:**
```bash
python AudiobookGenerator.py --project mybook --rvc-voice sigma_male_narrator
```

## Text Processing and Preprocessing

### Multi-Format Support

#### Enhanced PDF Processing
- **Hierarchical Navigation**: Chapter → Page → Word selection
- **Interactive Start Points**: Choose exact start location
- **Chapter Detection**: Bookmarks, text patterns, intelligent fallback

#### EPUB Processing  
- **Section-based extraction** with granular selection
- **Interactive start point selection**

#### Universal Text Cleaning
- **Phonetic pronunciation fixes** for all non-SSML engines
- **Configurable text preprocessing** per engine
- **Smart punctuation and abbreviation handling**

## Command Line Interface

### Core Usage Patterns

```bash
# Project lifecycle
python AudiobookGenerator.py --init mybook
python AudiobookGenerator.py --project mybook --input book.epub
python AudiobookGenerator.py --project mybook --tts-engine f5 --rvc-voice sigma_male_narrator

# Advanced features
python AudiobookGenerator.py --project mybook --interactive-start
python AudiobookGenerator.py --project mybook --sections 1 2 3
python AudiobookGenerator.py --project mybook --batch-name "quality-test"
```

### Dynamic Configuration Override

```bash
# Engine parameters can be overridden via CLI
python AudiobookGenerator.py --project mybook --bark-text-temp 0.2 --speed 1.2
```

## Configuration Snapshots and Analysis

### Job-Level Configuration Snapshots

**Every job creates:**
1. **Complete config snapshot**: `job/config.json` - Exact settings used
2. **Human-readable summary**: `job/config_summary.txt` - Easy to read
3. **Processing log**: `job/progress.log` - Timing and status info

**Benefits:**
- 🔍 **Compare results** - See exactly what settings produced each output
- 📊 **A/B testing** - Try different parameters and compare
- 🔄 **Reproducibility** - Copy any job's config to reproduce results
- 📈 **Optimization** - Identify best settings over time

### Example Config Summary
```txt
=== AUDIOBOOK GENERATION CONFIG SUMMARY ===
Generated: 2025-06-04 15:30:45

=== JOB METADATA ===
Project: mybook
Batch: complete
TTS Engine: f5
Sections: All

=== F5 ENGINE SETTINGS ===
speed: 0.9
cfg_strength: 1.5
nfe_step: 128
ref_audio: samples/my_voice.wav

=== RVC SETTINGS (sigma_male_narrator) ===
model: Sigma Male Narrator
n_semitones: -6
index_rate: 0.3
```

## Error Handling and Quality Control

### Enhanced Error Recovery

**Configurable retry logic:**
```json
{
  "bark": {
    "retry_failed_chunks": 3,
    "error_recovery_mode": "retry",
    "skip_failed_chunks": false
  }
}
```

**Quality Control Features:**
- **Artifact detection** with configurable thresholds
- **Voice consistency monitoring** 
- **Audio validation** and automatic correction
- **Progress tracking** with detailed logging

## Development Workflow

### Adding New Parameters

**Old Way (Required Code Changes):**
1. Edit engine Python file
2. Add parameter to hardcoded config dictionary
3. Add parameter handling in generation function
4. Test and deploy

**New Way (No Code Changes Required):**
1. Add parameter to `default_config.json`
2. Parameter automatically detected and used
3. Done!

### Testing and Optimization

1. **Create test project**:
   ```bash
   python AudiobookGenerator.py --init test-project
   ```

2. **Add test content and run**:
   ```bash
   python AudiobookGenerator.py --project test-project --tts-engine f5
   ```

3. **Compare config snapshots** to find optimal settings

4. **Update `default_config.json`** with best parameters for future projects

### Adding New TTS Engines

1. **Create engine file**: `new_engine.py`
2. **Use dynamic loading pattern**:
   ```python
   from engine_registry import extract_engine_config, create_generation_params
   
   def process_new_text_file(text_file, output_dir, config, paths):
       # Get ALL parameters from config automatically
       engine_config = extract_engine_config(config, 'new_engine', verbose=True)
       
       # Generate with dynamic parameters
       params = create_generation_params(base_params, engine_config, 
                                       filter_function=api.generate, verbose=True)
       result = api.generate(**params)
   ```

3. **Register engine**:
   ```python
   register_engine(name='new_engine', processor_func=process_new_text_file)
   ```

4. **Add to registry imports**

## Dependencies and Installation

### Core Dependencies
```bash
# Basic functionality
pip install pathlib datetime json

# Audio processing  
pip install scipy torchaudio

# Text processing
pip install beautifulsoup4 ebooklib pymupdf

# TTS Engines (optional)
pip install bark               # Bark TTS
pip install edge-tts          # Microsoft EdgeTTS (free)
pip install f5-tts            # F5-TTS
pip install TTS               # Coqui XTTS
```

### External Tools
- **RVC**: Requires `urvc` command-line tool in PATH
- **FFmpeg**: Required for audio processing

## Performance and Quality Optimization

### Memory Management
- **Automatic model reloading** (configurable per engine)
- **GPU memory clearing** between chunks
- **Garbage collection** with configurable frequency

### Quality Control
- **Configurable artifact detection** and removal
- **Voice consistency monitoring**
- **Audio normalization** and enhancement
- **Progress tracking** with detailed metrics

### Chunk Size Optimization
Each engine has optimized defaults but fully configurable:
- **Bark**: 150 chars (conservative for quality)
- **EdgeTTS**: 1000 chars (cloud service handles large chunks)
- **F5**: Single-pass processing (no chunking)
- **XTTS**: 400 chars (balanced for quality and speed)

## Future Enhancement Areas

### Planned Features
1. **Additional TTS Engines**: OpenAI TTS, Azure Speech Services (paid)
2. **Web Interface**: Browser-based project management
3. **Batch Processing**: Multiple projects simultaneously
4. **Advanced Analytics**: Quality metrics and optimization suggestions

### Architecture Improvements
1. **Plugin System**: Hot-swappable engine plugins
2. **Cloud Integration**: Cloud storage and processing options
3. **Real-time Monitoring**: Live progress and quality tracking
4. **Advanced Configuration**: GUI-based parameter tuning

## Migration from v1.0

### Key Changes
- ✅ **External configuration**: All parameters in JSON files
- ✅ **Dynamic loading**: No hardcoded engine parameters
- ✅ **Config snapshots**: Complete reproducibility
- ✅ **Enhanced quality**: Better error handling and optimization

### Migration Steps
1. **Replace engine files** with dynamic versions
2. **Update `engine_registry.py`** with enhanced version
3. **Create `default_config.json`** with desired parameters
4. **Existing projects** automatically updated on first run

---

## Quick Reference

### Essential Commands
```bash
# Project lifecycle
python AudiobookGenerator.py --init mybook
python AudiobookGenerator.py --project mybook --input book.epub  
python AudiobookGenerator.py --project mybook --tts-engine f5 --rvc-voice sigma_male_narrator

# Interactive features
python AudiobookGenerator.py --project mybook --interactive-start
python AudiobookGenerator.py --project mybook --list

# Quality optimization  
python AudiobookGenerator.py --project mybook --batch-name "test-run"
# Compare job snapshots in output/mybook/jobs/*/config.json
```

### File Locations
- **Master Config**: `default_config.json` (project root)
- **Project Config**: `output/project_name/config/config.json`
- **Job Snapshots**: `output/project_name/jobs/batch_name/config.json`
- **Final Audio**: `output/project_name/jobs/batch_name/batch_name_rvc.wav`
- **Config Summary**: `output/project_name/jobs/batch_name/config_summary.txt`

### Configuration Philosophy
- 📄 **JSON is truth** - All parameters externally controlled
- 🔍 **Dynamic detection** - Add any parameter, engine uses it automatically  
- 📸 **Complete snapshots** - Every job's exact settings preserved
- 🔄 **No code changes** - New features work by updating JSON only

This architecture enables rapid experimentation, precise reproducibility, and effortless optimization of audiobook generation quality.