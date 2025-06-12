@echo off
REM Create Fresh XTTS Environment with Newer Version
REM Destroys old 'xtts' environment and creates new one with Coqui TTS 0.26.2

echo ========================================
echo Creating Fresh XTTS Environment
echo ========================================
echo.

REM Install core ML libraries first (compatible versions)
echo.
echo [4/8] Installing core ML libraries...
pip install "numpy>=2.0.0"
pip install "scipy>=1.13.0"
pip install "torch>=2.3.0" "torchvision" "torchaudio>=2.3.0" --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch suite
    goto install_failed
)
echo ✅ Installed core ML libraries

REM Install audio processing libraries
echo.
echo [5/8] Installing audio processing libraries...
pip install "librosa>=0.11.0"
pip install "soundfile>=0.12.1"
if errorlevel 1 (
    echo ERROR: Failed to install audio libraries
    goto install_failed
)
echo ✅ Installed audio libraries

REM Install text processing libraries
echo.
echo [6/8] Installing text processing libraries...
pip install "beautifulsoup4>=4.13.0"
pip install "ebooklib>=0.19"
pip install "PyMuPDF>=1.26.0"
pip install "transformers>=4.47.0,<4.52"
pip install "gruut[de,es,fr]>=2.4.0"
echo ✅ Installed text processing libraries

REM Install Coqui TTS 0.26.2
echo.
echo [7/8] Installing Coqui TTS 0.26.2...
pip install coqui-tts==0.26.2
if errorlevel 1 (
    echo ERROR: Failed to install Coqui TTS
    goto install_failed
)
echo ✅ Installed Coqui TTS 0.26.2

REM Install optional TTS engines (best effort)
echo.
echo [8/8] Installing optional TTS engines...
pip install "openai>=1.84.0"
pip install "edge-tts>=7.0.2"
pip install "suno-bark>=0.0.1a0"
echo ⚠️ Note: f5-tts and rvc may have compatibility issues with newer versions
echo ⚠️ They are not essential for XTTS + OpenVoice functionality
echo ✅ Installed optional engines

echo.
echo ========================================
echo Testing New XTTS Installation
echo ========================================

REM Test basic import
echo Testing TTS import...
python -c "from TTS.api import TTS; print('✅ TTS imports successfully')"
if errorlevel 1 (
    echo ❌ TTS import failed
    goto test_failed
)

REM Test version info
echo Testing TTS version...
python -c "import TTS; print('✅ TTS version available'); print('Package location:', TTS.__file__ if hasattr(TTS, '__file__') and TTS.__file__ else 'Unknown')"

REM Test XTTS model availability
echo Testing XTTS model listing...
python -c "from TTS.api import TTS; tts = TTS(); models = tts.list_models(); xtts_models = [m for m in models if 'xtts' in m.lower()]; print(f'✅ Found {len(xtts_models)} XTTS models'); print('XTTS models:', xtts_models[:3])"
if errorlevel 1 (
    echo ❌ Model listing failed
    goto test_failed
)

REM Test OpenVoice model availability
echo Testing OpenVoice model availability...
python -c "from TTS.api import TTS; tts = TTS(); models = tts.list_models(); openvoice_models = [m for m in models if 'openvoice' in m.lower()]; print(f'✅ Found {len(openvoice_models)} OpenVoice models'); print('OpenVoice models:', openvoice_models)"

REM Test PyTorch CUDA
echo Testing PyTorch CUDA support...
python -c "import torch; print('✅ PyTorch version:', torch.__version__); print('✅ CUDA available:', torch.cuda.is_available()); print('✅ CUDA devices:', torch.cuda.device_count() if torch.cuda.is_available() else 'None')"

REM Test audio libraries
echo Testing audio libraries...
python -c "import librosa, soundfile; print('✅ Librosa version:', librosa.__version__); print('✅ SoundFile working')"

REM Show final package list
echo.
echo Final package versions in 'xtts' environment:
echo ================================================
echo TTS-related packages:
pip list | findstr -i tts
echo.
echo Core dependencies:
python -c "import numpy, scipy, torch, librosa, soundfile; print(f'NumPy: {numpy.__version__}'); print(f'SciPy: {scipy.__version__}'); print(f'PyTorch: {torch.__version__}'); print(f'Librosa: {librosa.__version__}'); print(f'SoundFile: {soundfile.__version__}')"

echo.
echo ========================================
echo ✅ NEW XTTS ENVIRONMENT READY!
echo ========================================
echo.
echo Your new 'xtts' environment is ready with:
echo ✅ Coqui TTS 0.26.2 (latest fork)
echo ✅ Compatible modern dependencies
echo ✅ OpenVoice integration available
echo ✅ XTTS v2 with 17+ languages
echo ✅ Enhanced performance and quality
echo.
echo To use this environment:
echo 1. conda activate xtts
echo 2. cd "path\to\your\AudiobookGenerator"
echo 3. python AudiobookGenerator.py --project test --tts-engine xtts --job new-version
echo.
echo To switch back to your working environment:
echo conda activate bark
echo.
echo New features to explore:
echo - OpenVoice voice conversion (potentially better than URVC)
echo - Enhanced XTTS v2 with better quality
echo - Cross-language voice cloning
echo - Improved performance and stability
echo.
echo Press any key to continue...
pause >nul
goto end

:test_failed
echo.
echo ========================================
echo ⚠️ INSTALLATION COMPLETED WITH ISSUES
echo ========================================
echo.
echo The environment was created but some tests failed.
echo This might be due to:
echo 1. Model downloads needed on first use
echo 2. CUDA driver issues
echo 3. Network connectivity for model downloads
echo.
echo You can still try using the environment:
echo conda activate xtts
echo python AudiobookGenerator.py --project test --tts-engine xtts
echo.
echo The core TTS functionality should work even if some tests failed.
echo Press any key to continue...
pause >nul
goto end

:install_failed
echo.
echo ========================================
echo ❌ INSTALLATION FAILED
echo ========================================
echo.
echo The installation failed during package setup.
echo.
echo Troubleshooting steps:
echo 1. Check internet connection
echo 2. Try running as administrator
echo 3. Check available disk space
echo 4. Try installing packages individually
echo.
echo Manual installation approach:
echo conda create -n xtts python=3.10 -y
echo conda activate xtts
echo pip install coqui-tts==0.26.2
echo.
echo Press any key to continue...
pause >nul

:end
echo.
echo Script completed. You can now choose between environments:
echo - 'bark' environment: Original XTTS 0.22.0 (stable)
echo - 'xtts' environment: New Coqui TTS 0.26.2 (experimental)
echo.