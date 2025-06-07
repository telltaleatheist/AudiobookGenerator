# AudiobookGenerator Development Documentation v2.2

## Overview

AudiobookGenerator is a sophisticated Python-based audiobook generation system that converts text documents (EPUB, PDF, TXT) into high-quality audiobooks using multiple TTS engines and voice conversion technology. The system features a **dynamic configuration architecture** where all settings are externally controlled via JSON configuration files, eliminating hardcoded parameters.

**NEW in v2.2**: **Centralized Config Manager** with no-defaults architecture and universal phrase-aware chunking for XTTS.

## TTS Engine Quality Rankings

### 1. **XTTS (Coqui)** - ⭐⭐⭐⭐⭐ BEST LOCAL MODEL
- **Quality**: Best local TTS model available
- **Requirements**: Requires RVC post-processing to clean output
- **Features**: Multilingual, voice cloning, advanced prosody control
- **Best For**: High-quality audiobooks with custom voices
- **Note**: Near-perfect when combined with URVC cleaning

### 2. **OpenAI TTS** - ⭐⭐⭐⭐ PREMIUM CLOUD
- **Quality**: Excellent cloud-based TTS
- **Cost**: Paid service (~$15/1M characters)
- **Features**: Multiple high-quality voices, consistent output
- **Best For**: Commercial projects with budget for cloud services
- **Note**: No voice cloning, but excellent built-in voices

### 3. **F5-TTS** - ⭐⭐⭐ GOOD LOCAL
- **Quality**: Good local TTS with voice cloning
- **Requirements**: Single reference audio file
- **Features**: Fast single-pass processing, decent voice cloning
- **Best For**: Quick voice cloning projects
- **Note**: Simpler than XTTS but lower quality output

### 4. **EdgeTTS (Microsoft)** - ⭐⭐ FREE CLOUD
- **Quality**: Decent cloud-based TTS
- **Cost**: Free (with usage limits)
- **Features**: Multiple voices, basic prosody control
- **Best For**: Testing and free projects
- **Note**: No voice cloning, limited customization

### 5. **Bark** - ⭐ FAIR LOCAL
- **Quality**: Fair quality, inconsistent output
- **Features**: Voice presets, some emotional control
- **Issues**: Prone to artifacts, inconsistent prosody
- **Best For**: Experimental projects only
- **Note**: Generally not recommended for production use

## URVC (Ultimate RVC) - Audio Enhancement System

### Quality Enhancement
- **Purpose**: Cleans up TTS output to professional quality
- **Impact**: Transforms mediocre TTS into high-quality speech
- **Default Model**: "Sigma Male Narrator" - top-tier voice conversion
- **Custom Training**: Train your own voice models for personalized output

### Why RVC is Essential
Nearly all TTS models produce raw output with:
- ❌ Inconsistent prosody
- ❌ Artifacts and glitches  
- ❌ Unnatural intonation
- ❌ Poor voice consistency

URVC post-processing delivers:
- ✅ Professional voice quality
- ✅ Consistent tone and prosody
- ✅ Artifact removal
- ✅ Natural speech patterns

## System Architecture

### Core Components

1. **Main Entry Point**: `AudiobookGenerator.py` - Command-line interface and orchestration
2. **Config Manager**: `config_manager.py` - **NEW**: Centralized configuration with no-defaults architecture
3. **Project Management**: `project_manager.py` - Handles project structure and file management
4. **Pipeline Management**: `pipeline_manager.py` - Orchestrates the 5-phase processing pipeline
5. **Dynamic Engine Registry**: `engine_registry.py` - Plugin system with dynamic parameter loading
6. **Text Preprocessing**: `preprocessing.py` + `preprocessing_pdf.py` - Text extraction and cleaning
7. **TTS Engines**: Individual engine implementations with universal phrase-aware chunking
8. **Audio Processing**: `audio_processor.py` - Audio combination, RVC conversion, and post-processing

### Centralized Configuration Management (NEW v2.2)

**Key Innovation**: NO configuration defaults in code. All defaults come from `default_config.json`.

#### Config Manager System

```bash
# Create default config template
python config_manager.py --create-default

# Create new project with config
python config_manager.py --create-project mybook

# Copy default config to existing project  
python config_manager.py --copy-to-project mybook

# Validate configuration file
python config_manager.py --validate output/mybook/config/config.json
```

#### No-Defaults Architecture

**Core Principle**: Eliminate caching issues and configuration inconsistencies.

- ✅ **Single source of truth**: `default_config.json` contains ALL defaults
- ✅ **Graceful failure**: Missing config values show helpful error messages  
- ✅ **No caching issues**: All settings loaded fresh from JSON files
- ✅ **Easy debugging**: Always know exactly what settings are being used

#### Error Handling
```python
# Engines now fail gracefully with helpful messages
ConfigError: Missing required XTTS configuration: temperature
💡 Check your config.json file and ensure all XTTS settings are present
💡 Run: python config_manager.py --create-default
```

### Processing Pipeline (5 Phases)

1. **Config Snapshot Creation**: Complete configuration captured before processing begins
2. **Preprocessing**: Extract and clean text from source documents with universal phrase preservation
3. **TTS Generation**: Convert text to speech using selected TTS engine with dynamic parameters
4. **Audio Combination**: Combine individual audio chunks with intelligent silence gaps
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
├── config_manager.py     # NEW: Centralized config management
```

## Installation and Setup

### Prerequisites
```bash
# Python 3.8+ required
python --version

# Install core dependencies
pip install pathlib datetime json scipy torchaudio beautifulsoup4 ebooklib pymupdf
```

### TTS Engine Installation

#### 1. XTTS (Recommended)
```bash
pip install TTS
# Test installation
python -c "from TTS.api import TTS; print('XTTS installed successfully')"
```

#### 2. OpenAI TTS (Paid)
```bash
pip install openai
# Set API key
export OPENAI_API_KEY="your-api-key-here"
```

#### 3. F5-TTS
```bash
pip install f5-tts
# Verify installation
python -c "import f5_tts; print('F5-TTS installed successfully')"
```

#### 4. EdgeTTS (Free)
```bash
pip install edge-tts
# Test installation
edge-tts --list-voices | head -5
```

#### 5. Bark (Optional)
```bash
pip install bark-tts
# Note: Large model downloads on first use
```

### URVC Installation (Essential for Quality)

#### Option 1: pip install (Recommended)
```bash
pip install urvc
# Verify installation
urvc --help
```

#### Option 2: Manual Installation
```bash
# Clone repository
git clone https://github.com/JarodMica/ultimate-rvc
cd ultimate-rvc
pip install -e .
```

#### Download RVC Models
```bash
# Download the default "Sigma Male Narrator" model
# Follow URVC documentation for model installation
# Models are typically placed in ~/.urvc/models/
```

### System Setup

1. **Create default configuration**:
   ```bash
   python config_manager.py --create-default
   ```

2. **Install FFmpeg** (required for audio processing):
   ```bash
   # Windows (using chocolatey)
   choco install ffmpeg
   
   # macOS
   brew install ffmpeg
   
   # Linux
   sudo apt install ffmpeg
   ```

3. **Create first project**:
   ```bash
   python AudiobookGenerator.py --init mybook
   ```

## Multi-Voice RVC System

### Architecture Overview

The system supports **unlimited RVC voice profiles** with automatic discovery and dynamic configuration management. Each voice profile has its own configuration section that can be independently managed.

### Configuration Structure

```json
{
  "rvc_global": {
    "speed_factor": 1.0,
    "f0_method": "crepe",
    "clean_voice": true,
    "clean_strength": 0.3,
    "autotune_voice": true
  },
  "rvc_my_voice": {
    "model": "my_voice",
    "n_semitones": -2,
    "index_rate": 0.35,
    "protect_rate": 0.15,
    "rms_mix_rate": 0.4
  },
  "rvc_sigma_male_narrator": {
    "model": "Sigma Male Narrator",
    "n_semitones": -4,
    "index_rate": 0.4,
    "protect_rate": 0.4,
    "rms_mix_rate": 0.5
  },
  "rvc_custom_voice": {
    "model": "Custom Voice Model",
    "n_semitones": 0,
    "index_rate": 0.45,
    "protect_rate": 0.3
  }
}
```

### Voice Management Features

#### Command Line Interface
```bash
# List available voices
python AudiobookGenerator.py --project mybook --list-rvc-voices

# Select specific voice
python AudiobookGenerator.py --project mybook --rvc-voice sigma_male_narrator

# Use global speed override
python AudiobookGenerator.py --project mybook --rvc-voice custom_voice --speed 1.1

# Legacy support (deprecated)
python AudiobookGenerator.py --project mybook --rvc-model sigma_male_narrator
```

#### Adding New Voices

Adding new RVC voices is completely **plug-and-play**:

1. **Add configuration section**:
```json
{
  "rvc_celebrity_voice": {
    "model": "Celebrity Voice Model",
    "n_semitones": 0,
    "index_rate": 0.3,
    "protect_rate": 0.25
  }
}
```

2. **Voice automatically available**:
```bash
python AudiobookGenerator.py --project mybook --rvc-voice celebrity_voice
```

## TTS Engine System

### Universal Phrase-Aware Chunking (NEW v2.2)

All engines now use intelligent chunking that prevents splitting technical terms and maintains natural speech flow:

- ✅ **Preserves technical phrases**: "electromagnetic signature" stays together
- ✅ **Respects dialogue boundaries**: Proper speaker transitions
- ✅ **Universal patterns**: Works with any text content
- ✅ **No dictionaries**: Uses grammatical patterns, not hardcoded terms

### Supported TTS Engines

#### 1. XTTS (`xtts_engine.py`) - ⭐⭐⭐⭐⭐ BEST
- **Type**: Local neural TTS with voice cloning
- **Quality**: Best local model available, requires RVC for perfection
- **Features**: Multilingual, multiple reference samples, advanced prosody control
- **NEW**: Universal phrase-aware chunking prevents splitting technical terms
- **Dynamic Parameters**: All XTTS API parameters automatically supported

#### 2. OpenAI TTS (`openai_engine.py`) - ⭐⭐⭐⭐ PREMIUM
- **Type**: Premium cloud TTS service
- **Quality**: Excellent professional voices
- **Features**: Multiple high-quality voices, consistent output
- **Cost**: ~$15 per 1M characters
- **Limitations**: No voice cloning capability

#### 3. F5-TTS (`f5_engine.py`) - ⭐⭐⭐ GOOD
- **Type**: Local voice cloning with reference audio
- **Features**: Single-pass processing, voice cloning with minimal samples
- **Quality**: Good voice cloning, faster than XTTS
- **Dynamic Parameters**: All F5-TTS API parameters automatically supported

#### 4. EdgeTTS (`edge_engine.py`) - ⭐⭐ FREE
- **Type**: Free Microsoft cloud TTS service
- **Features**: Multiple voices, basic prosody control
- **Limitations**: No voice cloning, limited customization
- **Cost**: Free with usage limits

#### 5. Bark (`bark_engine.py`) - ⭐ FAIR
- **Type**: Local neural TTS with voice presets
- **Quality**: Fair quality, prone to artifacts
- **Features**: Voice presets, some emotional control
- **Issues**: Inconsistent prosody, not recommended for production

### Engine Registration and Loading

```python
# Engines automatically register with no hardcoded config
def register_xtts_engine():
    register_engine(
        name='xtts',
        processor_func=process_xtts_text_file
        # NO default_config parameter needed!
    )
```

## URVC (Ultimate RVC) Commands Reference

### Main Commands

#### Voice Conversion
```bash
# Basic voice conversion
urvc generate convert-voice input.wav ./output/ "Sigma Male Narrator"

# Voice conversion with pitch shift (male to female)
urvc generate convert-voice input.wav ./output/ "My Voice" --n-octaves 1

# High-quality conversion with enhancement
urvc generate convert-voice input.wav ./output/ "Sigma Male Narrator" \
  --split-voice --clean-voice --autotune-voice \
  --f0-method crepe --clean-strength 0.3
```

#### Speech Generation Pipeline
```bash
# Full TTS + RVC pipeline (EdgeTTS -> RVC conversion)
urvc generate speech run-pipeline "Hello world" "Sigma Male Narrator"

# Pipeline with custom settings
urvc generate speech run-pipeline "Hello world" "My Voice" \
  --tts-voice en-US-JennyNeural \
  --f0-method crepe \
  --clean-speech \
  --output-format wav
```

### Voice Conversion Parameters

#### Required Arguments
- `voice_track` - Path to audio file to convert
- `directory` - Output directory for converted audio
- `model_name` - Name of RVC model to use

#### Main Options
```bash
--n-octaves INTEGER        # Octaves to pitch-shift (1 for male→female, -1 for female→male)
--n-semitones INTEGER      # Semi-tones to pitch-shift (fine-tuning)
--f0-method [rmvpe|crepe|crepe-tiny|fcpe]  # Pitch extraction method
--index-rate FLOAT         # Voice model influence rate (0-1, default: 0.3)
--rms-mix-rate FLOAT       # Volume envelope blending (0-1, default: 1.0)
--protect-rate FLOAT       # Consonant/breathing protection (0-0.5, default: 0.33)
--hop-length INTEGER       # CREPE pitch checking frequency (1-512, default: 128)
```

#### Voice Enhancement Options
```bash
--split-voice              # Split voice into segments for better quality
--autotune-voice          # Apply autotune to converted voice
--autotune-strength FLOAT # Autotune intensity (0-1, default: 1.0)
--clean-voice             # Apply noise reduction algorithms
--clean-strength FLOAT    # Cleaning intensity (0-1, default: 0.7)
```

### Training Your Own Voice Models

#### 1. Prepare Dataset
```bash
# Create dataset from audio files
urvc train populate-dataset my_voice /path/to/audio/files
```

#### 2. Preprocess Dataset
```bash
# Basic preprocessing
urvc train preprocess-dataset my_voice /path/to/audio/files

# High-quality dataset (clean studio recordings)
urvc train preprocess-dataset my_voice /path/to/audio/files \
  --sample-rate 48000 \
  --split-method Skip \
  --no-filter-audio \
  --no-clean-audio

# Low-quality dataset (phone recordings, background noise)
urvc train preprocess-dataset my_voice /path/to/audio/files \
  --sample-rate 40000 \
  --split-method Automatic \
  --filter-audio \
  --clean-audio \
  --clean-strength 0.8
```

#### 3. Extract Features
```bash
urvc train extract-features my_voice
```

#### 4. Train Model
```bash
urvc train run-training my_voice
```

#### Training Tips
- **Dataset Quality**: Higher quality input = better voice model
- **Duration**: Aim for 10+ minutes of clean speech per model
- **Consistency**: Use consistent recording conditions
- **Content**: Varied speech content improves model quality
- **File Format**: WAV files work best

## Voice Cloning and Reference Audio

### Automatic Sample Detection

The system automatically detects voice samples:

```
project/samples/
├── my_voice.wav           # Audio file
├── my_voice.txt          # Optional: transcript for F5-TTS
├── speaker2.wav          # Multiple samples supported by XTTS
└── speaker2.txt
```

**Detection Logic:**
- **XTTS**: Uses all `.wav` files (supports multiple references)
- **F5-TTS**: Uses first `.wav` + matching `.txt` file (auto-transcribe if no text)
- **OpenAI/Edge/Bark**: Use built-in voices (ignore samples directory)

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
- **Universal phrase preservation** prevents splitting technical terms

## Command Line Interface

### Core Usage Patterns

```bash
# Project lifecycle
python AudiobookGenerator.py --init mybook
python AudiobookGenerator.py --project mybook --input book.epub
python AudiobookGenerator.py --project mybook --tts-engine xtts --rvc-voice sigma_male_narrator

# Voice management
python AudiobookGenerator.py --project mybook --list-rvc-voices
python AudiobookGenerator.py --project mybook --rvc-voice custom_voice
python AudiobookGenerator.py --project mybook --rvc-voice sigma_male_narrator --speed 1.1

# Advanced features
python AudiobookGenerator.py --project mybook --interactive-start
python AudiobookGenerator.py --project mybook --sections 1 2 3
python AudiobookGenerator.py --project mybook --batch-name "quality-test"
```

### TTS Engine Selection

```bash
# XTTS (Best local model)
python AudiobookGenerator.py --project mybook --tts-engine xtts --rvc-voice sigma_male_narrator

# OpenAI (Premium cloud)
python AudiobookGenerator.py --project mybook --tts-engine openai --engine-voice nova

# F5-TTS (Good local)
python AudiobookGenerator.py --project mybook --tts-engine f5

# EdgeTTS (Free cloud)
python AudiobookGenerator.py --project mybook --tts-engine edge --engine-voice en-US-JennyNeural

# Bark (Fair local)
python AudiobookGenerator.py --project mybook --tts-engine bark --engine-voice v2/en_speaker_6
```

### Universal Voice Parameter

The `--engine-voice` parameter works with any TTS engine:

```bash
# OpenAI voices
--engine-voice nova
--engine-voice alloy
--engine-voice echo

# EdgeTTS voices  
--engine-voice en-US-JennyNeural
--engine-voice en-GB-SoniaNeural

# Bark voices
--engine-voice v2/en_speaker_6
--engine-voice v2/en_speaker_9

# XTTS built-in speakers (if available)
--engine-voice speaker_1
```

## Configuration Snapshots and Analysis

### Job-Level Configuration Snapshots

**Every job creates:**
1. **Complete config snapshot**: `job/config.json` - Exact settings used
2. **Human-readable summary**: `job/config_summary.txt` - Easy to read
3. **Processing log**: `job/progress.log` - Timing and status info

### Example Config Summary
```txt
=== AUDIOBOOK GENERATION CONFIG SUMMARY ===
Generated: 2025-06-05 15:30:45

=== JOB METADATA ===
Project: mybook
Batch: complete
TTS Engine: xtts
RVC Voice: sigma_male_narrator
Sections: All

=== XTTS ENGINE SETTINGS ===
temperature: 0.5
repetition_penalty: 7.0
top_k: 15
top_p: 0.6
chunk_max_chars: 249

=== RVC GLOBAL SETTINGS ===
speed_factor: 1.0
f0_method: crepe
clean_voice: true

=== RVC VOICE SETTINGS (sigma_male_narrator) ===
model: Sigma Male Narrator
n_semitones: -4
index_rate: 0.4
protect_rate: 0.4
```

## Development Workflow

### Configuration Changes (No Code Required)

1. **Edit `default_config.json`** - Add any new parameters
2. **Copy to project**: `python config_manager.py --copy-to-project mybook`  
3. **Test immediately** - Parameters automatically detected and used

### Engine Modifications (Rare)

```python
# NEW pattern for engines
from config_manager import ConfigManager, ConfigError

def process_engine(config):
    config_manager = ConfigManager()
    
    try:
        required_fields = ['temperature', 'model_name', 'speed']
        for field in required_fields:
            if field not in config['engine_name']:
                raise ConfigError(f"Missing required configuration: {field}")
    except ConfigError as e:
        print(f"❌ Configuration Error: {e}")
        print(f"💡 Run: python config_manager.py --create-default")
        return []
```

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

### Adding New RVC Voices

**Super Simple Process:**
1. **Add config section**:
```json
{
  "rvc_new_voice": {
    "model": "New Voice Model",
    "n_semitones": 0,
    "index_rate": 0.4
  }
}
```

2. **Voice immediately available**:
```bash
python AudiobookGenerator.py --project mybook --rvc-voice new_voice
```

### Testing and Optimization

1. **Create test project**:
   ```bash
   python AudiobookGenerator.py --init test-project
   ```

2. **Add test content and run**:
   ```bash
   python AudiobookGenerator.py --project test-project --tts-engine xtts --rvc-voice sigma_male_narrator
   ```

3. **Compare config snapshots** to find optimal settings

4. **Update `default_config.json`** with best parameters for future projects

## Troubleshooting

### Configuration Issues
```bash
# Missing config values
❌ ConfigError: Missing required XTTS configuration: temperature
✅ Solution: python config_manager.py --create-default

# Cache issues with old configs  
❌ Settings not updating between runs
✅ Solution: Config manager eliminates caching - settings load fresh each time

# RVC voice not found
❌ RVC voice 'my_voice' not found!  
✅ Solution: python AudiobookGenerator.py --project mybook --list-rvc-voices
```

### TTS Engine Issues
```bash
# XTTS installation
❌ ModuleNotFoundError: No module named 'TTS'
✅ Solution: pip install TTS

# OpenAI API key
❌ OpenAI API key not found
✅ Solution: export OPENAI_API_KEY="your-key-here"

# EdgeTTS network issues
❌ Connection timeout
✅ Solution: Check internet connection, try again
```

### RVC Issues
```bash
# URVC not found
❌ Command 'urvc' not found
✅ Solution: pip install urvc

# RVC model missing
❌ Model 'Sigma Male Narrator' not found
✅ Solution: Download and install RVC models

# Audio quality issues
❌ Output sounds robotic
✅ Solution: Adjust protect_rate, index_rate, and clean_strength
```

### Audio Quality Issues
```bash
# High-pitched dialogue
✅ Lower temperature: 0.5, top_p: 0.6, top_k: 15

# Technical term mispronunciation  
✅ Universal chunking algorithm now preserves technical phrases

# Long pauses between chunks
✅ Adjust silence gaps: silence_gap_sentence: 0.3, silence_gap_dramatic: 0.45

# Robotic RVC output
✅ Adjust RVC settings: protect_rate: 0.4, index_rate: 0.3, clean_strength: 0.3
```

## Performance and Quality Optimization

### Recommended Settings by Engine

#### XTTS (Best Quality)
```json
{
  "xtts": {
    "temperature": 0.5,
    "repetition_penalty": 7.0,
    "top_k": 15,
    "top_p": 0.6,
    "chunk_max_chars": 249,
    "gpt_cond_len": 12
  }
}
```

#### RVC Post-Processing (Essential)
```json
{
  "rvc_sigma_male_narrator": {
    "model": "Sigma Male Narrator",
    "n_semitones": -4,
    "f0_method": "crepe",
    "index_rate": 0.4,
    "protect_rate": 0.4,
    "clean_voice": true,
    "clean_strength": 0.3,
    "autotune_voice": true,
    "autotune_strength": 0.3
  }
}
```

### Memory Management
- **Automatic model reloading** (configurable per engine)
- **GPU memory clearing** between chunks
- **Garbage collection** with configurable frequency

### Quality Control
- **Universal phrase preservation** prevents splitting technical terms
- **Intelligent silence gaps** based on content context
- **Voice consistency monitoring**
- **Audio normalization** and enhancement

## Quick Reference

### Essential Commands
```bash
# Setup
python config_manager.py --create-default
python AudiobookGenerator.py --init mybook

# Basic processing (recommended)
python AudiobookGenerator.py --project mybook --input book.epub  
python AudiobookGenerator.py --project mybook --tts-engine xtts --rvc-voice sigma_male_narrator

# Voice management
python AudiobookGenerator.py --project mybook --list-rvc-voices
python AudiobookGenerator.py --project mybook --rvc-voice custom_voice --speed 1.1

# Quality testing
python AudiobookGenerator.py --project mybook --batch-name "test-run"
# Compare job snapshots in output/mybook/jobs/*/config.json
```

### File Locations
- **Master Config**: `default_config.json` (project root)
- **Config Manager**: `config_manager.py` (project root)
- **Project Config**: `output/project_name/config/config.json`
- **Job Snapshots**: `output/project_name/jobs/batch_name/config.json`
- **Final Audio**: `output/project_name/jobs/batch_name/batch_name_rvc.wav`

### Configuration Philosophy
- 📄 **JSON is truth** - All parameters externally controlled
- 🚫 **No defaults in code** - Everything comes from config files
- 🔍 **Dynamic detection** - Add any parameter, engine uses it automatically  
- 📸 **Complete snapshots** - Every job's exact settings preserved
- 🔄 **No caching issues** - Settings load fresh each time
- 🎭 **Voice flexibility** - Unlimited RVC voices with plug-and-play setup

### Recommended Workflow
1. **Start with XTTS + Sigma Male Narrator** for best quality
2. **Test with short text** to dial in settings
3. **Use config snapshots** to compare results
4. **Scale to full audiobooks** once settings are optimized
5. **Train custom RVC models** for personalized voices

This architecture enables rapid experimentation, precise reproducibility, and effortless optimization of audiobook generation quality with professional-grade voice conversion.