# AudiobookGenerator Development Documentation v3.0

## Overview

AudiobookGenerator is a sophisticated Python-based audiobook generation system that converts text documents (EPUB, PDF, TXT) into high-quality audiobooks using multiple TTS engines and voice conversion technology. The system features a **section-based, resumable pipeline architecture** with dynamic configuration management where all settings are externally controlled via JSON configuration files.

**NEW in v3.0**: **Section-based processing pipeline** with resumable progress, simplified engines with dynamic parameter loading, improved thermal management for sustained AI workloads, and comprehensive error handling.

## System Requirements

### Software Dependencies

#### Core System
- **Python**: 3.10.16
- **PyTorch**: 2.2.1+cu121 (CUDA 12.1)
- **CUDA**: 12.1 (GPU acceleration required for optimal performance)
- **FFmpeg**: 7.1.1 (audio processing)

#### Core Python Libraries
- **NumPy**: 1.22.0
- **SciPy**: 1.11.4  
- **torchaudio**: 2.2.1+cu121
- **librosa**: 0.10.0 (audio processing)
- **soundfile**: 0.12.1 (audio I/O)

#### Text Processing
- **beautifulsoup4**: 4.13.4 (HTML/EPUB parsing)
- **EbookLib**: 0.19 (EPUB processing)
- **PyMuPDF**: 1.26.0 (PDF processing)

#### TTS Engines
- **TTS (Coqui XTTS)**: 0.22.0
- **openai**: 1.84.0
- **f5-tts**: 1.1.5
- **edge-tts**: 7.0.2
- **suno-bark**: 0.0.1a0

#### Voice Conversion
- **URVC (Ultimate RVC)**: Latest version (command-line interface)
- **rvc (base package)**: 0.3.5 (underlying dependency)

### Hardware Requirements

#### Minimum System
- **GPU**: NVIDIA GPU with 6GB+ VRAM (CUDA compatible)
- **RAM**: 16GB system RAM
- **Storage**: 20GB+ free space for models and temporary files

#### Recommended System  
- **GPU**: NVIDIA RTX 3080+ with 10GB+ VRAM
- **RAM**: 32GB+ system RAM
- **Storage**: SSD with 50GB+ free space
- **Cooling**: Adequate case ventilation for sustained AI workloads

#### Thermal Considerations
⚠️ **Critical**: Sustained TTS + RVC processing generates significant heat. Ensure:
- GPU temperatures stay below 80°C during long runs
- Adequate case cooling (especially important for nighttime processing)
- Monitor with `nvidia-smi` during extended sessions
- Consider processing in smaller batches if overheating occurs

## TTS Engine Quality Rankings (Updated Based on Testing)

### 1. **XTTS (Coqui)** - ⭐⭐⭐⭐⭐ BEST LOCAL MODEL
- **Quality**: Best local TTS model available
- **Voice Cloning**: Excellent with single reference sample
- **Requirements**: Requires RVC post-processing for professional quality
- **Features**: Low-level API access, advanced prosody control, multilingual
- **Processing**: Section-based with intelligent chunking (no phrase splitting)
- **Best For**: High-quality audiobooks with custom voices
- **Note**: Near-perfect quality when combined with URVC post-processing

### 2. **OpenAI TTS** - ⭐⭐⭐⭐ PREMIUM CLOUD
- **Quality**: Excellent cloud-based TTS with consistent output
- **Cost**: ~$15 per 1M characters (cost estimation built-in)
- **Features**: Multiple high-quality voices, fast processing
- **Limitations**: No voice cloning capability
- **Best For**: Commercial projects with budget for cloud services
- **Note**: Excellent built-in voices, no additional processing needed

### 3. **F5-TTS** - ⭐⭐⭐ GOOD LOCAL WITH VOICE CLONING
- **Quality**: Good local TTS with voice cloning capabilities
- **Voice Cloning**: Single reference audio file with auto-transcription
- **Features**: Fast single-pass processing, decent voice cloning
- **Processing**: Can handle full sections or chunked processing
- **Best For**: Quick voice cloning projects
- **Note**: Simpler setup than XTTS but lower overall quality

### 4. **EdgeTTS (Microsoft)** - ⭐⭐ FREE CLOUD
- **Quality**: Decent cloud-based TTS with multiple voices
- **Cost**: Free (with usage limits and rate limiting)
- **Features**: Multiple voices, basic prosody control, async processing
- **Limitations**: No voice cloning, limited customization
- **Best For**: Testing, free projects, or backup engine
- **Note**: Good for validation but not production quality

### 5. **Bark** - ⭐ FAIR LOCAL (SIMPLIFIED)
- **Quality**: Fair quality with simplified implementation  
- **Features**: Voice presets, basic pronunciation fixes
- **Issues**: Prone to artifacts, inconsistent prosody
- **Implementation**: Simplified version with essential features only
- **Best For**: Experimental projects or specific voice requirements
- **Note**: Not recommended for production audiobooks

## URVC (Ultimate RVC) - Professional Voice Enhancement

### Quality Enhancement System
- **Purpose**: Transforms raw TTS output into professional-quality speech
- **Impact**: Essential for achieving broadcasting-quality audiobooks
- **Default Model**: "Sigma Male Narrator" - premium voice conversion model
- **Custom Training**: Support for training personalized voice models

### Why URVC is Essential
Raw TTS output typically suffers from:
- ❌ Inconsistent prosody and intonation
- ❌ Audio artifacts and glitches  
- ❌ Unnatural speech patterns
- ❌ Poor voice consistency across long content

URVC post-processing delivers:
- ✅ Professional broadcasting quality
- ✅ Consistent tone and prosody throughout
- ✅ Artifact removal and audio enhancement
- ✅ Natural speech flow and rhythm
- ✅ Thermal management during processing

### Thermal Management for URVC
⚠️ **Important**: URVC processing is GPU-intensive and generates significant heat:
- Monitor GPU temperatures during long sessions
- Process smaller batches if temperatures exceed 80°C
- Ensure adequate cooling, especially during nighttime processing
- Consider air conditioning or additional case fans for extended runs

## System Architecture (Section-Based Pipeline)

### Core Philosophy: Section-Based Processing

**Key Innovation**: Process audiobooks in manageable ~30-minute sections rather than as monolithic files.

#### Benefits:
- ✅ **Resumable**: Can restart from any failed section
- ✅ **Memory Efficient**: Prevents GPU memory exhaustion
- ✅ **Thermal Friendly**: Natural cooling breaks between sections
- ✅ **Progress Tracking**: Clear progress indicators
- ✅ **Quality Control**: Can re-process individual sections if needed

### Processing Pipeline (3 Phases)

#### Phase 1: Preprocessing and Section Creation
- Extract and clean text from source documents
- Apply engine-specific text preprocessing (pronunciation fixes, etc.)
- Split text into ~30-minute sections (configurable)
- Save section files for processing

#### Phase 2: Section Processing Loop
For each section:
1. **TTS Generation**: Convert section text to audio chunks
2. **Audio Combination**: Combine chunks with intelligent silence gaps
3. **RVC Processing**: Apply voice conversion using URVC
4. **Master Combination**: Add completed section to final audiobook
5. **Progress Update**: Mark section complete for resume capability

#### Phase 3: Cleanup
- Remove temporary files
- Finalize output audio
- Generate completion report

### File Structure (Clean Organization)

```
output/
├── project_name/
│   ├── source/           # Input files (.epub, .pdf, .txt)
│   ├── samples/          # Voice reference audio for cloning
│   │   ├── sample1.wav   # Reference audio file
│   │   └── sample1.txt   # Optional transcript (F5-TTS)
│   ├── config/           # Project configuration
│   │   └── config.json   # Project settings
│   ├── jobs/             # Processing batches
│   │   └── batch_name/
│   │       ├── clean_text.txt           # Preprocessed text
│   │       ├── config.json              # Job config snapshot
│   │       ├── progress.log             # Processing progress
│   │       ├── project_batch_master.wav # Final output
│   │       ├── sections/                # Organized section files
│   │       │   ├── section_001.txt      # Section text
│   │       │   ├── section_001_tts.wav  # TTS output
│   │       │   └── section_001_rvc.wav  # RVC output
│   │       └── temp_files/              # Temporary processing files
│   └── README.md
```

### Core Components

1. **Main Entry Point**: `AudiobookGenerator.py` - Command-line interface
2. **Pipeline Manager**: `pipeline_manager.py` - Orchestrates section-based processing
3. **Section Manager**: `section_manager.py` - Smart text splitting with duration estimation
4. **Project Manager**: `project_manager.py` - Project structure and configuration
5. **Engine Registry**: `engines/__init__.py` - Dynamic engine loading and registration
6. **Individual Engines**: Simplified implementations with dynamic parameter loading
7. **Audio Processing**: `audio_combiner.py`, `rvc_processor.py` - Audio handling
8. **Text Processing**: `text_processor.py`, `file_extractors.py` - Text extraction and cleaning

### Configuration Management (No-Defaults Architecture)

**Core Principle**: All configuration parameters stored in JSON files, no hardcoded defaults in code.

#### Config File Hierarchy
1. **Default Config**: `default_config.json` - Master template
2. **Project Config**: `output/project/config/config.json` - Project-specific settings  
3. **Job Config**: `output/project/jobs/[job]/config.json` - Complete snapshot of job settings

#### Dynamic Parameter Loading
```python
# Engines automatically detect and use ANY parameter from config
engine_config = extract_engine_config(config, 'xtts', verbose=True)
# All parameters automatically passed to TTS generation function
generation_params = create_generation_params(base_params, engine_config, filter_function=tts_func)
```

## Installation and Setup

### Prerequisites
```bash
# Verify Python version
python --version  # Should be 3.10+

# Install CUDA toolkit if not present
# https://developer.nvidia.com/cuda-downloads
```

### Core Dependencies
```bash
# Install core audio and text processing libraries
pip install scipy==1.11.4 torchaudio==2.2.1 beautifulsoup4==4.13.4 
pip install ebooklib==0.19 PyMuPDF==1.26.0 librosa==0.10.0 soundfile==0.12.1
```

### TTS Engine Installation

#### 1. XTTS (Recommended for Best Quality)
```bash
pip install TTS==0.22.0
# Test installation
python -c "from TTS.api import TTS; print('XTTS installed successfully')"
```

#### 2. OpenAI TTS (Premium Cloud Option)
```bash
pip install openai==1.84.0
# Set API key
export OPENAI_API_KEY="your-api-key-here"
```

#### 3. F5-TTS (Good Local Alternative)
```bash
pip install f5-tts==1.1.5
# Verify installation
python -c "from f5_tts.api import F5TTS; print('F5-TTS installed successfully')"
```

#### 4. EdgeTTS (Free Testing Option)
```bash
pip install edge-tts==7.0.2
# Test installation
edge-tts --list-voices | head -5
```

#### 5. Bark (Optional/Experimental)
```bash
pip install suno-bark==0.0.1a0
# Note: Large model downloads on first use
```

### URVC Installation (Essential for Professional Quality)

#### Installation
```bash
# Install URVC (Ultimate RVC)
# Follow installation instructions from URVC repository
# Typically installed via git clone + pip install
```

#### Verify Installation
```bash
urvc --help
# Should show URVC command options
```

#### Download Voice Models
- Download "Sigma Male Narrator" model (recommended default)
- Place models in appropriate URVC models directory
- Test with sample audio file

### FFmpeg Installation
```bash
# Windows (using chocolatey)
choco install ffmpeg

# macOS
brew install ffmpeg

# Linux
sudo apt install ffmpeg

# Verify installation
ffmpeg -version  # Should show version 7.1.1+
```

### System Setup

1. **Create default configuration**:
   ```bash
   python AudiobookGenerator.py --create-default-config
   ```

2. **Create first project**:
   ```bash
   python AudiobookGenerator.py --init mybook
   ```

3. **Add source material and voice samples**:
   ```bash
   # Copy book file to output/mybook/source/
   # Copy voice sample to output/mybook/samples/
   ```

## Command Line Interface

### Project Lifecycle

#### Initial Setup
```bash
# Create new project
python AudiobookGenerator.py --init mybook

# Add source file
python AudiobookGenerator.py --project mybook --input book.epub

# Add voice samples (copy to output/mybook/samples/)
```

#### Basic Processing (Recommended)
```bash
# Best quality: XTTS + URVC
python AudiobookGenerator.py --project mybook --tts-engine xtts --rvc-voice sigma_male_narrator

# Free option: EdgeTTS only
python AudiobookGenerator.py --project mybook --tts-engine edge --skip-rvc

# Cloud option: OpenAI TTS
python AudiobookGenerator.py --project mybook --tts-engine openai --engine-voice nova
```

#### Advanced Options
```bash
# Process specific sections
python AudiobookGenerator.py --project mybook --tts-engine xtts --sections 1 2 3

# Custom job naming
python AudiobookGenerator.py --project mybook --tts-engine xtts --job "high-quality-test"

# Skip RVC processing
python AudiobookGenerator.py --project mybook --tts-engine xtts --skip-rvc

# Resume failed job (automatic - just re-run same command)
python AudiobookGenerator.py --project mybook --tts-engine xtts --job "failed-job"
```

### Voice Management

#### RVC Voice Selection
```bash
# List available RVC voices
python AudiobookGenerator.py --project mybook --list-rvc-voices

# Use specific RVC voice
python AudiobookGenerator.py --project mybook --rvc-voice sigma_male_narrator

# Use custom trained voice
python AudiobookGenerator.py --project mybook --rvc-voice my_custom_voice
```

#### TTS Engine Voice Selection
```bash
# OpenAI voices
python AudiobookGenerator.py --project mybook --tts-engine openai --engine-voice nova
python AudiobookGenerator.py --project mybook --tts-engine openai --engine-voice alloy

# EdgeTTS voices
python AudiobookGenerator.py --project mybook --tts-engine edge --engine-voice en-US-JennyNeural
python AudiobookGenerator.py --project mybook --tts-engine edge --engine-voice en-GB-SoniaNeural

# Bark voices
python AudiobookGenerator.py --project mybook --tts-engine bark --engine-voice v2/en_speaker_6
```

## Voice Cloning and Reference Audio

### Automatic Sample Detection

The system automatically detects voice samples from the `samples/` directory:

```
project/samples/
├── my_voice.wav           # Primary reference audio
├── my_voice.txt          # Optional transcript (F5-TTS)
├── speaker2.wav          # Additional samples (XTTS)
└── speaker2.txt          # Optional transcript
```

#### Engine-Specific Behavior:
- **XTTS**: Uses all `.wav` files (supports multiple references for better quality)
- **F5-TTS**: Uses first `.wav` file + matching `.txt` file (auto-transcribe if no text file)
- **OpenAI/Edge/Bark**: Use built-in voices (ignore samples directory)

### Voice Sample Quality Guidelines

#### For Best Results:
- **Duration**: 30 seconds to 2 minutes of clear speech
- **Quality**: Studio-quality recordings preferred
- **Content**: Varied speech content (not just single words)
- **Format**: WAV format, 44.1kHz or higher sample rate
- **Clarity**: No background noise, echo, or artifacts

#### XTTS Multiple Samples:
- Use 2-5 different audio samples for best voice cloning
- Samples should show different emotions/speaking styles
- Total combined duration: 2-10 minutes

## Configuration System

### Multi-Voice RVC System

The system supports unlimited RVC voice profiles with automatic discovery:

```json
{
  "rvc_global": {
    "speed_factor": 1.0,
    "f0_method": "crepe",
    "clean_voice": true,
    "clean_strength": 0.3,
    "autotune_voice": true
  },
  "rvc_sigma_male_narrator": {
    "model": "Sigma Male Narrator",
    "n_semitones": -2,
    "index_rate": 0.4,
    "protect_rate": 0.4,
    "rms_mix_rate": 0.5,
    "split_voice": true,
    "autotune_strength": 0.3
  },
  "rvc_custom_voice": {
    "model": "My Custom Voice",
    "n_semitones": 0,
    "index_rate": 0.35,
    "protect_rate": 0.25,
    "rms_mix_rate": 0.4
  }
}
```

### Adding New RVC Voices

Adding new voices is completely plug-and-play:

1. **Add configuration section**:
```json
{
  "rvc_new_voice": {
    "model": "New Voice Model Name",
    "n_semitones": 0,
    "index_rate": 0.4,
    "protect_rate": 0.3,
    "rms_mix_rate": 0.5
  }
}
```

2. **Voice automatically available**:
```bash
python AudiobookGenerator.py --project mybook --rvc-voice new_voice
```

### Dynamic Parameter System

All engines support dynamic parameter loading - any parameter in the config file is automatically detected and used:

```json
{
  "xtts": {
    "temperature": 0.65,
    "repetition_penalty": 5.5,
    "top_k": 15,
    "top_p": 0.75,
    "speed": 0.98,
    "any_new_parameter": "automatically_used"
  }
}
```

## Performance Optimization

### Recommended Settings by Engine

#### XTTS (Optimal Quality)
```json
{
  "xtts": {
    "temperature": 0.65,
    "repetition_penalty": 5.5,
    "top_k": 15,
    "top_p": 0.75,
    "chunk_max_chars": 250,
    "reload_model_every_chunks": 3,
    "speed": 0.98,
    "gpt_cond_len": 60,
    "max_ref_len": 64
  }
}
```

#### RVC Post-Processing (Essential for Quality)
```json
{
  "rvc_sigma_male_narrator": {
    "model": "Sigma Male Narrator",
    "n_semitones": -2,
    "f0_method": "crepe",
    "index_rate": 0.4,
    "protect_rate": 0.4,
    "rms_mix_rate": 0.5,
    "clean_voice": true,
    "clean_strength": 0.3,
    "autotune_voice": true,
    "autotune_strength": 0.3
  }
}
```

### Thermal Management Strategies

#### For Long Audiobooks (5+ hours):
1. **Monitor temperatures**: Use `nvidia-smi` to watch GPU temps
2. **Process in sections**: Let system cool between sections
3. **Nighttime considerations**: Ensure adequate cooling without AC
4. **Job processing**: Process 3-5 sections, pause, repeat

#### Temperature Thresholds:
- **Safe**: Under 75°C
- **Caution**: 75-80°C (monitor closely)
- **Danger**: Over 80°C (reduce workload, improve cooling)

#### Cooling Solutions:
- Additional case fans
- Undervolting GPU slightly
- Room air conditioning or fans
- Processing during cooler parts of day

## Troubleshooting

### Common Issues and Solutions

#### Configuration Errors
```bash
# Missing config sections
❌ ConfigError: Missing required XTTS configuration: temperature
✅ Solution: Check config.json file, ensure all engine sections present

# Invalid parameter values
❌ Temperature outside valid range
✅ Solution: Check parameter ranges in engine documentation
```

#### TTS Engine Issues
```bash
# XTTS installation problems
❌ ModuleNotFoundError: No module named 'TTS'
✅ Solution: pip install TTS==0.22.0

# OpenAI API issues
❌ OpenAI API key not found
✅ Solution: export OPENAI_API_KEY="your-key-here"

# F5-TTS model loading
❌ F5TTS.__init__() got unexpected keyword argument
✅ Solution: Check F5-TTS version compatibility
```

#### Thermal Issues (Critical)
```bash
# GPU overheating during processing
❌ Error code 3221225477 (ACCESS_VIOLATION)
❌ Driver crashes during long runs
❌ Performance degradation over time

✅ Solutions:
1. Monitor GPU temps with nvidia-smi
2. Ensure adequate case cooling
3. Process smaller batches (--sections 1 2 3)
4. Add room cooling for nighttime processing
5. Consider undervolting GPU
6. Take cooling breaks between long sessions
```

#### Audio Quality Issues
```bash
# Robotic/artificial speech
✅ Adjust XTTS temperature (try 0.5-0.7)
✅ Modify RVC settings: protect_rate: 0.4, index_rate: 0.3

# Long pauses between sentences
✅ Adjust audio.silence_gap settings in config

# Technical terms mispronounced
✅ Add pronunciation fixes to text preprocessing

# Inconsistent voice quality
✅ Ensure RVC processing is enabled
✅ Check voice sample quality
✅ Verify model loading properly
```

#### Processing Errors
```bash
# Section processing failures
❌ RVC processing failed for section X
✅ Solution: Check URVC installation and model availability
✅ Resume processing - pipeline will skip completed sections

# Memory issues
❌ CUDA out of memory
✅ Solution: Reduce chunk_max_chars, enable model reloading

# File access errors  
❌ Permission denied on temp files
✅ Solution: Run with appropriate permissions, check disk space
```

### Resume Capability

The section-based pipeline automatically resumes from failures:

```bash
# If processing fails at section 15/50
# Simply re-run the same command:
python AudiobookGenerator.py --project mybook --tts-engine xtts --job same-job

# Pipeline will:
# ✅ Skip sections 1-14 (already complete)
# ✅ Resume from section 15
# ✅ Continue to completion
```

## Performance Benchmarks

### Typical Processing Times (per section ~30 minutes audio):

#### XTTS + URVC (Recommended):
- **TTS Generation**: 15-25 minutes
- **RVC Processing**: 5-10 minutes  
- **Total per section**: 20-35 minutes

#### OpenAI TTS (Cloud):
- **TTS Generation**: 2-5 minutes
- **No RVC needed**: 0 minutes
- **Total per section**: 2-5 minutes

#### Processing Full Audiobook (10 hours):
- **XTTS + URVC**: 6-12 hours total processing time
- **OpenAI TTS**: 1-2 hours total processing time
- **Note**: Times vary based on text complexity and hardware

### Hardware Impact:
- **RTX 3080 (10GB)**: Baseline performance
- **RTX 4080+ (16GB+)**: 20-30% faster
- **Lower VRAM (6-8GB)**: May require smaller chunks, longer processing

## Development Workflow

### Adding New Parameters (No Code Required)

1. **Add to config file**:
```json
{
  "engine_name": {
    "existing_param": "value",
    "new_parameter": "new_value"
  }
}
```

2. **Parameter automatically detected and used** - no code changes needed!

### Engine Modifications (Rare)

All engines follow the same pattern:
```python
def process_engine_text_file(text_file: str, output_dir: str, config: Dict[str, Any], paths: Dict[str, Any]) -> List[str]:
    # Extract dynamic config
    engine_config = extract_engine_config(config, 'engine_name', verbose=True)
    
    # Validate required parameters
    required_params = ['param1', 'param2']
    missing_params = validate_required_params(engine_config, required_params, 'engine_name')
    if missing_params:
        return []
    
    # All config parameters automatically available in engine_config
    # Process text and return generated files
```

### Testing Workflow

1. **Create test project**:
   ```bash
   python AudiobookGenerator.py --init test-project
   ```

2. **Add short test content** (1-2 pages)

3. **Test different engines**:
   ```bash
   python AudiobookGenerator.py --project test-project --tts-engine xtts --job xtts-test
   python AudiobookGenerator.py --project test-project --tts-engine edge --job edge-test
   ```

4. **Compare quality and adjust settings** in config files

5. **Scale to full content** once settings optimized

## Quick Reference

### Essential Commands
```bash
# Setup
python AudiobookGenerator.py --init mybook
python AudiobookGenerator.py --project mybook --input book.epub

# Best quality processing
python AudiobookGenerator.py --project mybook --tts-engine xtts --rvc-voice sigma_male_narrator

# Free processing 
python AudiobookGenerator.py --project mybook --tts-engine edge --skip-rvc

# Resume failed processing
python AudiobookGenerator.py --project mybook --tts-engine xtts --job same-name

# Monitor thermal during processing
nvidia-smi
```

### File Locations
- **Final Audio**: `output/project/jobs/[job]/project_batch_master.wav`
- **Section Files**: `output/project/jobs/[job]/sections/`
- **Config Snapshot**: `output/project/jobs/[job]/config.json`
- **Progress Log**: `output/project/jobs/[job]/progress.log`

### Thermal Management Commands
```bash
# Monitor GPU temperature
nvidia-smi

# Check GPU usage and temperature continuously
watch -n 2 nvidia-smi

# Process in small batches if overheating
python AudiobookGenerator.py --project mybook --sections 1 2 3
# Wait for cooling, then:
python AudiobookGenerator.py --project mybook --sections 4 5 6
```

### Quality Optimization Workflow
1. **Start with XTTS + Sigma Male Narrator** for best quality
2. **Test with 1-2 sections** to optimize settings  
3. **Monitor temperatures** during initial runs
4. **Use section-based processing** for thermal management
5. **Scale to full audiobooks** once settings confirmed
6. **Train custom RVC models** for personalized voices

This architecture enables high-quality audiobook generation with professional voice conversion while maintaining system stability through intelligent thermal and memory management.