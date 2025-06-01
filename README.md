# AudiobookGenerator
A comprehensive text-to-speech audiobook generation system supporting multiple TTS engines, voice cloning, and advanced audio processing.

## Features
Multiple TTS Engines: Bark, EdgeTTS, F5-TTS, XTTS v2
Voice Cloning: RVC (Retrieval-based Voice Conversion) support
Project Management: Organized project structure with batch processing
Interactive Start Points: Hierarchical chapter/page/word selection for PDFs
Multiple File Formats: EPUB, PDF, TXT support
Advanced Audio Processing: Automatic silence removal, speed adjustment, voice enhancement
## Quick Start
bash
# Create a new project
python AudiobookGenerator.py --init mybook

# Add source file
python AudiobookGenerator.py --project mybook --input book.epub

# Generate audiobook with Bark
python AudiobookGenerator.py --project mybook --tts-engine bark

# Process specific chapters/sections
python AudiobookGenerator.py --project mybook --chapters 1 2 3

# Interactive start point selection
python AudiobookGenerator.py --project mybook --interactive-start
## Tested Environment Versions
This project has been tested with the following versions:

## Core Dependencies
Python: 3.10+
PyTorch: 2.2.1+cu121
NumPy: 1.22.0
SciPy: 1.11.4
LibROSA: 0.10.0
SoundFile: 0.12.1
CUDA: 12.6 (Driver 560.94)
## TTS Engine Versions
edge-tts: 7.0.2
f5-tts: 1.1.5
TTS (XTTS): 0.22.0
bark: (No version info - install from git)
## Check Your Installation
bash
# Quick version verification
python -c "
import sys, torch, numpy, scipy, librosa, soundfile, TTS
print('Python:', sys.version.split()[0])
print('PyTorch:', torch.__version__)
print('NumPy:', numpy.__version__)
print('SciPy:', scipy.__version__)
print('LibROSA:', librosa.__version__)
print('SoundFile:', soundfile.__version__)
print('TTS/XTTS:', TTS.__version__)
print('CUDA Available:', torch.cuda.is_available())
"

# Check TTS engine versions
pip show edge-tts f5-tts TTS
## Installation Guide
1. Install Anaconda
## Windows
Download Anaconda from anaconda.com
Run the installer as administrator
Choose "Add Anaconda to PATH" during installation
Restart your command prompt/PowerShell
Verify installation: conda --version
## macOS
Download Anaconda from anaconda.com
Open the downloaded .pkg file
Follow the installation wizard
Restart Terminal
Verify installation: conda --version
2. Why Use Conda Environments?
Environments are crucial for AI projects because:

Dependency Isolation: Different models require different package versions
Conflict Prevention: Avoid version conflicts between projects
Easy Switching: Switch between different model setups instantly
Reproducibility: Share exact environment configurations
System Protection: Keep your base Python installation clean
Without environments, you'll encounter:

Package version conflicts
Broken installations
Difficult debugging
System-wide package pollution
3. Setting Up AI Environments
## Create Base Environment
bash
# Create environment with Python 3.10 (recommended for AI models)
conda create -n audiobook-ai python=3.10 -y
conda activate audiobook-ai

# Install common dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy librosa soundfile
## Create Specialized Environments
bash
# For Bark TTS
conda create -n bark python=3.10 -y
conda activate bark
pip install git+https://github.com/suno-ai/bark.git

# For RVC
conda create -n rvc python=3.10 -y
conda activate rvc
pip install ultimate-rvc

# For F5-TTS
conda create -n f5 python=3.10 -y
conda activate f5
pip install f5-tts
## Model Installation
## Core AudiobookGenerator Dependencies
bash
conda activate audiobook-ai

# Install PyTorch with CUDA support (tested versions)
pip install torch==2.2.1+cu121 torchaudio==2.2.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Core audio processing
pip install numpy==1.22.0 scipy==1.11.4
pip install librosa==0.10.0 soundfile==0.12.1

# Document processing
pip install ebooklib beautifulsoup4  # EPUB support
pip install pymupdf                   # PDF support

# Or install all at once from requirements
pip install -r requirements.txt
## 1. Ultimate RVC
bash
conda activate rvc
pip install ultimate-rvc

# Verify installation
urvc --help
urvc-web  # Launch web interface
RVC Model Setup:

Place your trained RVC models in the appropriate directory
Models should include .pth and .index files
Use urvc-web for easy model management
## 2. Bark TTS
bash
conda activate bark
pip install git+https://github.com/suno-ai/bark.git

# Install matching dependencies from your working environment
pip install torch==2.2.1+cu121 torchaudio==2.2.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.22.0 scipy==1.11.4

# Verify installation (Note: Bark doesn't report version info)
python -c "from bark import generate_audio, SAMPLE_RATE; print('Bark installed successfully')"
Bark Features:

High-quality neural TTS
Multiple speaker voices (v2/en_speaker_0 to v2/en_speaker_9)
Supports emotional and expressive speech
## 3. XTTS v2
bash
conda activate audiobook-ai
pip install TTS==0.22.0

# Install compatible PyTorch
pip install torch==2.2.1+cu121 torchaudio==2.2.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "from TTS.api import TTS; print('XTTS v0.22.0 installed successfully')"
XTTS Features:

Multilingual support
Voice cloning with short audio samples
Real-time voice conversion
## 4. EdgeTTS (Microsoft)
bash
conda activate audiobook-ai
pip install edge-tts==7.0.2

# List available voices
edge-tts --list-voices | grep "en-US"

# Verify installation
python -c "import edge_tts; print('EdgeTTS installed successfully')"
EdgeTTS Features:

High-quality Microsoft voices
Fast processing
No GPU required
Free to use
## 5. F5-TTS
bash
conda activate f5
pip install f5-tts==1.1.5

# Install compatible PyTorch version
pip install torch==2.2.1+cu121 torchaudio==2.2.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "from f5_tts.api import F5TTS; print('F5-TTS v1.1.5 installed successfully')"
F5-TTS Features:

State-of-the-art quality
Voice cloning capabilities
Flow-matching based architecture
## 6. Engine Registry System
Your AudiobookGenerator uses a modular engine registry system that automatically detects and registers available TTS engines. The engines are loaded dynamically, so you only need to install the ones you plan to use.

## Essential Tools & Dependencies
## FFmpeg (Required for Audio Processing)
AudiobookGenerator uses FFmpeg for:

Combining audio chunks with silence gaps
Speed adjustment (atempo filter)
Audio format conversion
Silence removal processing
## Windows Installation:

Download FFmpeg from ffmpeg.org
Choose "Windows builds" → "Windows builds by BtbN"
Download the latest release (ffmpeg-master-latest-win64-gpl.zip)
Extract to C:\ffmpeg
Add C:\ffmpeg\bin to your system PATH:
Press Win+R, type sysdm.cpl
Advanced → Environment Variables
Under System Variables, find "Path" → Edit
Add C:\ffmpeg\bin
Click OK and restart command prompt
Verify: ffmpeg -version
## macOS Installation:

bash
# Using Homebrew (recommended)
brew install ffmpeg

# Verify installation
ffmpeg -version
## Linux Installation:

bash
# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg

# CentOS/RHEL
sudo yum install ffmpeg
## Audio Processing Tools
bash
# Core audio processing (used by your audio_processor.py)
pip install scipy soundfile

# For more advanced audio operations
pip install librosa pydub

# For vocal separation with RVC (optional)
pip install spleeter
## PDF Processing
bash
# For PDF text extraction
pip install pymupdf

# Alternative PDF library
pip install pypdf2
## EPUB Processing
bash
# For EPUB file support
pip install ebooklib beautifulsoup4
## Project Structure
output/
├── project_name/
│   ├── source/           # Input files (.epub, .pdf, .txt)
│   ├── samples/          # Voice reference samples
│   ├── config/           # Configuration files
│   └── jobs/             # Processing outputs
│       └── batch_name/
│           ├── temp_files/
│           ├── combined.wav
│           └── final.wav
## Usage Examples
## Basic Workflow
bash
# 1. Create project
python AudiobookGenerator.py --init "my-audiobook"

# 2. Add source file
python AudiobookGenerator.py --project "my-audiobook" --input "book.epub"

# 3. Add voice samples (for F5/XTTS)
# Place .wav files in output/my-audiobook/samples/

# 4. Generate with preferred engine
python AudiobookGenerator.py --project "my-audiobook" --tts-engine f5
## Advanced Options
bash
# Custom batch name
python AudiobookGenerator.py --project mybook --batch-name "chapter-1" --chapters 1

# Skip RVC processing
python AudiobookGenerator.py --project mybook --skip-rvc

# Interactive start point
python AudiobookGenerator.py --project mybook --interactive-start

# Voice overrides
python AudiobookGenerator.py --project mybook --voice "en-US-AriaNeural" --rvc-model "my_voice"
## RVC Voice Conversion
bash
# Convert existing audio to your voice
urvc generate convert-voice "input.wav" "output_dir" "model_name" \
  --n-semitones -2 \
  --index-rate 1.0 \
  --protect-rate 0.4 \
  --rms-mix-rate 0.1 \
  --f0-method crepe \
  --clean-voice \
  --clean-strength 0.7 \
  --autotune-voice \
  --autotune-strength 0.4

  best settings: urvc generate convert-voice "C:\Users\tellt\OneDrive\Documents\tts\Audiobook Generator\output\eisenhauer\jobs\complete\Eisenhauer, Jay. ARC Genesis. 2025.wav" "C:\Users\tellt\OneDrive\Documents\tts\Audiobook Generator\output\eisenhauer\jobs\complete" my_voice --n-semitones -2 --index-rate 1.0 --protect-rate 0.4 --rms-mix-rate 0.1 --f0-method crepe --clean-voice --clean-strength 0.7 --hop-length 64 --split-voice --autotune-voice --autotune-strength 0.4

## Configuration
The system uses JSON configuration files with engine-specific settings:

json
{
  "tts_engine": "f5",
  "f5": {
    "model_name": "F5TTS_Base",
    "ref_audio": "path/to/sample.wav",
    "speed": 1.0
  },
  "rvc": {
    "model": "my_voice",
    "speed_factor": 1.0,
    "clean_silence": true
  }
}
## Troubleshooting
## Common Issues
## GPU Memory Errors:

bash
# Monitor GPU usage
nvidia-smi -l 1

# Reduce batch sizes in config files
# Use CPU fallback for some engines
## Environment Conflicts:

bash
# Always activate correct environment
conda activate audiobook-ai

# Check active environment
conda info --envs

# Verify versions match tested configuration
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import numpy; print('NumPy:', numpy.__version__)"
## Module Not Found:

bash
# Ensure you're in the right environment
conda activate <environment_name>

# Reinstall missing packages
pip install <package_name>
## Performance Tips
Use WAV files for best RVC quality
Separate environments for each TTS engine
Monitor GPU memory when processing long texts
Use batch processing for multiple chapters
Clean temp files between large projects
## Contributing
Fork the repository
Create a feature branch
Make your changes
Test with different TTS engines
Submit a pull request
## License
[Your License Here]

## Support
Issues: Submit GitHub issues for bugs
Discussions: Use GitHub discussions for questions
Documentation: Check the wiki for detailed guides
Note: This project requires significant computational resources. A modern GPU with 8GB+ VRAM is recommended for optimal performance.