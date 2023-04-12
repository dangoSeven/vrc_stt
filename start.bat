@echo off

echo Checking if FFmpeg is installed and added to the system PATH...
where /q ffmpeg
if errorlevel 1 (
    echo FFmpeg not found in system PATH. Please install FFmpeg and add it to your system PATH.
    pause
    exit /b
) else (
    echo FFmpeg found in system PATH.
)

echo Installing Torch with CUDA support...
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117

echo Installing dependencies from requirements.txt...
pip install -r requirements.txt

echo Running sttts.py...
python stt_vad.py