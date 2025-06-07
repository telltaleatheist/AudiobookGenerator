# AudiobookGenerator v3.0

A professional-grade audiobook generation system that converts text documents (EPUB, PDF, TXT) into high-quality audiobooks using multiple TTS engines and advanced voice conversion technology. Features a **section-based, resumable pipeline** with intelligent thermal management for sustained AI workloads.

---

## 🌟 Key Features

- **Section-Based Processing**: ~30-minute sections with resumable progress and thermal management
- **Multiple TTS Engines**: XTTS, OpenAI TTS, F5-TTS, EdgeTTS, and Bark with dynamic parameter loading
- **Professional Voice Conversion**: URVC (Ultimate RVC) post-processing for studio-quality output
- **Voice Cloning**: Create audiobooks with custom voices using reference audio samples
- **Multi-Format Support**: EPUB, PDF, and TXT files with intelligent text extraction
- **Smart Text Processing**: Prevents splitting technical terms and preserves phrase integrity
- **Project Management**: Organized workflow with batch processing and complete configuration snapshots
- **Thermal Management**: Built-in cooling strategies for extended processing sessions

---

## 🏆 TTS Engine Quality Rankings (Updated 2025)

### Local Models
- ⭐⭐⭐⭐⭐ **XTTS (Coqui)** - Best local model with excellent voice cloning (requires URVC)
- ⭐⭐⭐ **F5-TTS** - Good local voice cloning with fast single-pass processing
- ⭐ **Bark** - Fair quality, experimental use only (simplified implementation)

### Cloud Services  
- ⭐⭐⭐⭐ **OpenAI TTS** - Premium cloud service with excellent built-in voices (~$15/1M chars)
- ⭐⭐ **EdgeTTS (Microsoft)** - Free cloud service with decent quality and rate limits

💡 **Recommended**: XTTS + URVC for best audiobook quality, or OpenAI TTS for fast cloud processing

---

## 🚀 Quick Start

```bash
# 1. Setup and create project  
python AudiobookGenerator.py --init mybook

# 2. Add your book and voice sample
# - Copy book.epub to output/mybook/source/
# - Copy voice.wav to output/mybook/samples/ (for voice cloning)

# 3. Generate audiobook (best quality)
python AudiobookGenerator.py --project mybook --tts-engine xtts --rvc-voice sigma_male_narrator

# 4. Or use free cloud option
python AudiobookGenerator.py --project mybook --tts-engine edge --skip-rvc
```

---

## 📋 System Requirements

### Hardware (Recommended)
- **GPU**: NVIDIA RTX 3080+ with 10GB+ VRAM
- **RAM**: 32GB+ system memory
- **Storage**: SSD with 50GB+ free space
- **Cooling**: Adequate case ventilation for extended AI processing

### Software
- **Python**: 3.10.16+
- **PyTorch**: 2.2.1+cu121 (CUDA 12.1)
- **FFmpeg**: 7.1.1+ for audio processing
- **URVC**: Latest version for voice conversion

⚠️ **Thermal Management**: GPU temperatures should stay below 80°C during extended processing. Monitor with `nvidia-smi` and ensure adequate cooling.

---

## 🔧 Installation

### Core Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch==2.2.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121

# Install core libraries
pip install scipy==1.11.4 beautifulsoup4==4.13.4 ebooklib==0.19 
pip install PyMuPDF==1.26.0 librosa==0.10.0 soundfile==0.12.1
```

### TTS Engines (Install as needed)

```bash
# XTTS (recommended for best quality)
pip install TTS==0.22.0

# F5-TTS (good local alternative)  
pip install f5-tts==1.1.5

# EdgeTTS (free cloud option)
pip install edge-tts==7.0.2

# OpenAI TTS (premium cloud)
pip install openai==1.84.0
export OPENAI_API_KEY="your-api-key-here"

# Bark (experimental)
pip install suno-bark==0.0.1a0
```

### URVC Voice Conversion (Essential for Professional Quality)

```bash
# Install URVC (Ultimate RVC)
# Follow URVC installation guide from repository
# Verify installation:
urvc --help
```

### FFmpeg Installation

```bash
# Windows (chocolatey)
choco install ffmpeg

# macOS  
brew install ffmpeg

# Linux
sudo apt install ffmpeg

# Verify: ffmpeg -version (should show 7.1.1+)
```

---

## 🎯 Recommended Workflow

### For Best Results:
1. **Start with XTTS + Sigma Male Narrator**: Highest quality combination
2. **Test with short content first**: 1-2 sections to dial in settings  
3. **Monitor temperatures**: Use `nvidia-smi` during long runs
4. **Use section-based processing**: Natural cooling breaks, resumable progress
5. **Compare configuration snapshots**: Find optimal settings for your content
6. **Scale to full audiobooks**: Once settings are optimized

### Thermal Management Strategy:
- Process during cooler parts of day or with adequate room cooling
- Use smaller section batches if temperatures exceed 80°C
- Monitor GPU temps continuously during extended runs
- Take cooling breaks between processing sessions

---

## 📁 Project Structure (Clean Organization)

```
output/
├── project_name/
│   ├── source/           # Input files (.epub, .pdf, .txt)
│   ├── samples/          # Voice reference audio for cloning
│   │   ├── voice.wav     # Reference audio file
│   │   └── voice.txt     # Optional transcript (F5-TTS)
│   ├── config/           # Project configuration
│   │   └── config.json   # Project settings
│   └── jobs/             # Processing batches
│       └── batch_name/
│           ├── clean_text.txt           # Preprocessed text
│           ├── config.json              # Complete settings snapshot
│           ├── progress.log             # Processing progress  
│           ├── project_batch_master.wav # Final output
│           └── sections/                # Organized section files
│               ├── section_001.txt      # Section text
│               ├── section_001_tts.wav  # TTS output
│               └── section_001_rvc.wav  # RVC output
```

---

## 🎛️ Advanced Features

### Section-Based Processing with Resume Capability

```bash
# Process specific sections
python AudiobookGenerator.py --project mybook --sections 1 2 3

# Automatic resume from failures (just re-run same command)
python AudiobookGenerator.py --project mybook --tts-engine xtts --batch-name same-name
# ✅ Skips completed sections, resumes from failure point
```

### Multi-Voice RVC System

```bash
# List available RVC voices
python AudiobookGenerator.py --project mybook --list-rvc-voices

# Use specific voice with speed adjustment
python AudiobookGenerator.py --project mybook --rvc-voice custom_voice --speed 1.1

# Skip RVC for testing
python AudiobookGenerator.py --project mybook --tts-engine xtts --skip-rvc
```

### Voice Cloning Setup

```
project/samples/
├── my_voice.wav           # Primary reference (30s-2min recommended)
├── my_voice.txt           # Optional transcript for F5-TTS
├── speaker2.wav           # Additional samples (XTTS supports multiple)
└── speaker2.txt           # Optional transcript
```

**Voice Sample Guidelines**:
- **Duration**: 30 seconds to 2 minutes of clear speech
- **Quality**: Studio recordings preferred, no background noise
- **Content**: Varied speech content showing different emotions
- **Format**: WAV format, 44.1kHz+ sample rate

---

## ⚙️ Configuration System (Dynamic Parameter Loading)

AudiobookGenerator v3.0 features dynamic parameter detection - any parameter in config files is automatically used:

### Unlimited RVC Voices (Plug-and-Play)
```json
{
  "rvc_sigma_male_narrator": {
    "model": "Sigma Male Narrator",
    "n_semitones": -2,
    "index_rate": 0.4,
    "protect_rate": 0.4
  },
  "rvc_custom_voice": {
    "model": "My Custom Voice",
    "n_semitones": 0,
    "index_rate": 0.35
  }
}
```

### Engine Parameters (All Automatically Detected)
```json
{
  "xtts": {
    "temperature": 0.65,
    "repetition_penalty": 5.5,
    "top_k": 15,
    "top_p": 0.75,
    "any_new_parameter": "automatically_used"
  }
}
```

---

## 🎵 Audio Quality Optimization

### Recommended XTTS Settings (Tested Optimal)
```json
{
  "xtts": {
    "temperature": 0.65,
    "repetition_penalty": 5.5,
    "top_k": 15,
    "top_p": 0.75,
    "chunk_max_chars": 250,
    "speed": 0.98,
    "reload_model_every_chunks": 3
  }
}
```

### Recommended RVC Settings (Professional Quality)
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

---

## 🔥 Thermal Management

### Critical for Extended Processing:

```bash
# Monitor GPU temperature continuously
nvidia-smi

# Watch temperatures during processing  
watch -n 2 nvidia-smi

# Process in manageable batches if overheating
python AudiobookGenerator.py --project mybook --sections 1 2 3
# Wait for cooling, then continue:
python AudiobookGenerator.py --project mybook --sections 4 5 6
```

### Temperature Guidelines:
- **Safe**: Under 75°C
- **Caution**: 75-80°C (monitor closely)
- **Danger**: Over 80°C (stop processing, improve cooling)

### Common Error Code:
- **3221225477 (ACCESS_VIOLATION)**: Usually indicates GPU overheating/driver crash
- **Solution**: Improve cooling, process smaller batches, monitor temperatures

---

## 📖 Command Reference

### Basic Commands
```bash
# Create project
python AudiobookGenerator.py --init mybook

# Add source file
python AudiobookGenerator.py --project mybook --input book.epub

# Best quality processing
python AudiobookGenerator.py --project mybook --tts-engine xtts --rvc-voice sigma_male_narrator

# Free cloud processing
python AudiobookGenerator.py --project mybook --tts-engine edge --skip-rvc

# Premium cloud processing
python AudiobookGenerator.py --project mybook --tts-engine openai --engine-voice nova
```

### TTS Engine Selection
```bash
# XTTS with voice cloning
python AudiobookGenerator.py --project mybook --tts-engine xtts

# F5-TTS with voice cloning
python AudiobookGenerator.py --project mybook --tts-engine f5

# OpenAI with specific voice
python AudiobookGenerator.py --project mybook --tts-engine openai --engine-voice alloy

# EdgeTTS with specific voice
python AudiobookGenerator.py --project mybook --tts-engine edge --engine-voice en-US-JennyNeural
```

---

## 📊 Performance Benchmarks

### Typical Processing Times (per ~30-minute section):

| Engine | TTS Generation | RVC Processing | Total Time |
|--------|---------------|----------------|------------|
| **XTTS + URVC** | 15-25 min | 5-10 min | 20-35 min |
| **OpenAI TTS** | 2-5 min | N/A | 2-5 min |
| **F5-TTS + URVC** | 10-15 min | 5-10 min | 15-25 min |
| **EdgeTTS** | 3-8 min | N/A | 3-8 min |

### Full Audiobook (10 hours content):
- **XTTS + URVC**: 6-12 hours processing time
- **OpenAI TTS**: 1-2 hours processing time
- **Note**: Times vary based on hardware and text complexity

---

## 🔧 Tested Environment

### Current Stable Versions:
- **Python**: 3.10.16
- **PyTorch**: 2.2.1+cu121
- **CUDA**: 12.1
- **NumPy**: 1.22.0
- **SciPy**: 1.11.4

### TTS Engines:
- **TTS (XTTS)**: 0.22.0
- **openai**: 1.84.0
- **f5-tts**: 1.1.5
- **edge-tts**: 7.0.2
- **suno-bark**: 0.0.1a0

### Audio Processing:
- **FFmpeg**: 7.1.1
- **librosa**: 0.10.0
- **soundfile**: 0.12.1
- **torchaudio**: 2.2.1+cu121

### Voice Conversion:
- **URVC**: Latest version
- **rvc (base)**: 0.3.5

---

## 📖 Documentation

- **[development.md](development.md)** - Complete technical documentation
- **Engine Comparison** - Detailed TTS engine analysis and benchmarks
- **Configuration Reference** - All available parameters and settings
- **Voice Training Guide** - Create custom RVC models
- **Troubleshooting** - Common issues, thermal management, error codes

---

## 💡 Why AudiobookGenerator v3.0?

### Professional Quality
- **URVC Post-Processing**: Transforms raw TTS into studio-quality speech
- **Voice Cloning**: Custom voices with minimal reference audio
- **Section-Based Pipeline**: Prevents memory issues and enables progress tracking

### Production Ready
- **Resumable Processing**: Never lose progress on long audiobooks
- **Thermal Management**: Built-in protections for sustained AI workloads  
- **Configuration Snapshots**: Exact reproducibility of results
- **Dynamic Parameters**: Add new settings without code changes

### Flexible & Reliable
- **No Vendor Lock-in**: Works with local models, optional cloud services
- **Universal Chunking**: Preserves technical terms and phrase integrity
- **Multi-Engine Support**: Choose the best engine for your needs
- **Quality Control**: Automatic artifact detection and correction

---

## 🆘 Troubleshooting Quick Reference

### Common Issues:
```bash
# GPU overheating (Error 3221225477)
✅ Monitor with nvidia-smi, improve cooling, process smaller batches

# Out of memory errors
✅ Reduce chunk_max_chars, enable model reloading

# Poor audio quality
✅ Use XTTS + URVC, adjust temperature (0.5-0.7), verify voice samples

# Processing failures
✅ Check progress.log, resume with same command (automatic)

# RVC model not found
✅ Verify URVC installation and model availability
```

### Performance Tips:
- Use SSD storage for temp files
- Close unnecessary applications during processing
- Process during cooler ambient temperatures
- Consider room air conditioning for extended sessions

---

## 🤝 Contributing

- Fork the repository and create feature branches
- Test with multiple TTS engines and content types  
- Include thermal testing for extended processing
- Submit pull requests with detailed descriptions

---

## 🆘 Support

- **Issues**: Submit GitHub issues for bugs and feature requests
- **Documentation**: Check `development.md` for comprehensive guides
- **Discussions**: Use GitHub discussions for questions and community support

---

> ⚡ **Performance Note**: A modern NVIDIA GPU with 8GB+ VRAM is recommended for optimal performance. Monitor GPU temperatures during extended processing sessions and ensure adequate cooling for sustained AI workloads.

> 🎯 **Quality Note**: For professional audiobook quality, use XTTS + URVC combination. OpenAI TTS provides excellent quality without post-processing for cloud-based workflows.