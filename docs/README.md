# AudiobookGenerator

A professional-grade audiobook generation system that converts text documents (EPUB, PDF, TXT) into high-quality audiobooks using multiple TTS engines and advanced voice conversion technology.

---

## 🌟 [Key Features](#key-features)

- **Multiple TTS Engines**: XTTS, OpenAI TTS, F5-TTS, EdgeTTS, and Bark
- **Professional Voice Conversion**: URVC (Ultimate RVC) post-processing for studio-quality output
- **Voice Cloning**: Create audiobooks with custom voices using reference audio
- **Multi-Format Support**: EPUB, PDF, and TXT files
- **Smart Text Processing**: Universal phrase-aware chunking preserves technical terms
- **Project Management**: Organized workflow with batch processing and configuration snapshots

---

## 🏆 [TTS Engine Quality Rankings](#tts-engine-quality-rankings)

- ⭐⭐⭐⭐⭐ XTTS (Coqui) - Best local model with voice cloning
- ⭐⭐⭐⭐ OpenAI TTS - Premium cloud service with excellent quality
- ⭐⭐⭐ F5-TTS - Good local voice cloning with fast processing
- ⭐⭐ EdgeTTS (Microsoft) - Free cloud service with decent quality
- ⭐ Bark - Fair quality, experimental use only

💡 **Pro Tip**: Use XTTS + URVC post-processing for the best audiobook quality!

---

## 🚀 [Quick Start](#quick-start)

```bash
# 1. Setup and create project
python config_manager.py --create-default
python AudiobookGenerator.py --init mybook

# 2. Add your book
python AudiobookGenerator.py --project mybook --input book.epub

# 3. Generate audiobook (recommended setup)
python AudiobookGenerator.py --project mybook --tts-engine xtts --rvc-voice sigma_male_narrator
```

---

## 📋 [Prerequisites](#prerequisites)

- Python 3.10+ with PyTorch
- CUDA GPU (8GB+ VRAM recommended for best performance)
- FFmpeg for audio processing
- URVC for professional voice conversion

---

## 🔧 [Installation](#installation)

### Core Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install AudiobookGenerator dependencies
pip install pathlib datetime json scipy beautifulsoup4 ebooklib pymupdf

# Install FFmpeg (varies by OS - see development.md for details)
```

### TTS Engines (Install as needed)

```bash
# XTTS (recommended)
pip install TTS

# F5-TTS
pip install f5-tts

# EdgeTTS (free)
pip install edge-tts

# OpenAI TTS
pip install openai
export OPENAI_API_KEY="your-api-key-here"

# Bark (experimental)
pip install bark-tts
```

### URVC Voice Conversion (Essential for Quality)

```bash
pip install urvc

# Verify installation
urvc --help
```

---

## 🎯 [Recommended Workflow](#recommended-workflow)

- Start with **XTTS + Default Voice**: Best quality out of the box
- Test with **Short Content**: Fine-tune settings before full processing
- Use **Configuration Snapshots**: Compare results and optimize
- Train **Custom RVC Models**: For personalized voices (optional)

---

## 📁 [Project Structure](#project-structure)

```
output/
├── project_name/
│   ├── source/           # Input files (.epub, .pdf, .txt)
│   ├── samples/          # Voice reference audio for cloning
│   ├── config/           # Project configuration
│   └── jobs/             # Processing outputs with config snapshots
│       └── batch_name/
│           ├── config.json         # Complete settings snapshot
│           ├── config_summary.txt  # Human-readable summary
│           ├── batch_name_tts.wav  # TTS output
│           └── batch_name_rvc.wav  # Final RVC processed audio
```

---

## 🎛️ [Advanced Features](#advanced-features)

### Multi-Voice RVC System

```bash
# List available voices
python AudiobookGenerator.py --project mybook --list-rvc-voices

# Use specific voice with speed adjustment
python AudiobookGenerator.py --project mybook --rvc-voice custom_voice --speed 1.1
```

### Interactive Processing

```bash
# Choose exact start point in PDFs
python AudiobookGenerator.py --project mybook --interactive-start

# Process specific sections
python AudiobookGenerator.py --project mybook --sections 1 2 3
```

### Voice Cloning Setup

```
project/samples/
├── my_voice.wav           # Reference audio (10+ seconds recommended)
├── my_voice.txt           # Optional transcript for F5-TTS
└── speaker2.wav           # Multiple samples supported by XTTS
```

---

## ⚙️ [Configuration System](#configuration-system)

AudiobookGenerator v2.2 features a centralized configuration manager with no hardcoded defaults:

```bash
# Create/update configurations
python config_manager.py --create-default
python config_manager.py --copy-to-project mybook
python config_manager.py --validate output/mybook/config/config.json
```

All settings are externally controlled via JSON files, ensuring reproducible results and easy optimization.

---

## 🎵 [Audio Quality Optimization](#audio-quality-optimization)

### Recommended XTTS Settings

```json
{
  "xtts": {
    "temperature": 0.5,
    "repetition_penalty": 7.0,
    "top_k": 15,
    "top_p": 0.6,
    "chunk_max_chars": 249
  }
}
```

### Recommended RVC Settings

```json
{
  "rvc_sigma_male_narrator": {
    "model": "Sigma Male Narrator",
    "n_semitones": -4,
    "f0_method": "crepe",
    "index_rate": 0.4,
    "protect_rate": 0.4,
    "clean_voice": true,
    "clean_strength": 0.3
  }
}
```

---

## 📖 [Documentation](#documentation)

- **Complete Development Guide** - In-depth technical documentation
- **Engine Comparison** - Detailed TTS engine analysis
- **Configuration Reference** - All available parameters
- **Voice Training Guide** - Create custom RVC models
- **Troubleshooting** - Common issues and solutions

---

## 🔧 [Tested Environment](#tested-environment)

- Python: 3.10+
- PyTorch: 2.2.1+cu121
- CUDA: 12.6 (Driver 560.94)
- TTS (XTTS): 0.22.0
- F5-TTS: 1.1.5
- EdgeTTS: 7.0.2

---

## 💡 [Why AudiobookGenerator?](#why-audiobookgenerator)

- **Professional Quality**: URVC post-processing transforms raw TTS into studio-quality speech
- **Flexible Pipeline**: Mix and match TTS engines with voice conversion models
- **Reproducible Results**: Complete configuration snapshots for every generation
- **No Vendor Lock-in**: Works with local models - no cloud dependencies required
- **Voice Consistency**: Universal phrase-aware chunking prevents audio artifacts

---

## 🤝 [Contributing](#contributing)

- Fork the repository
- Create a feature branch
- Test with multiple TTS engines
- Submit a pull request

---

## 📄 [License](#license)

[Your License Here]

---

## 🆘 [Support](#support)

- **Issues**: Submit GitHub issues for bugs
- **Documentation**: Check `development.md` for comprehensive guides
- **Discussions**: Use GitHub discussions for questions

> ⚡ **Performance Note**: A modern GPU with 8GB+ VRAM is recommended for optimal performance with local TTS models.
