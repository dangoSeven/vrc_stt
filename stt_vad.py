import pyaudio
import wave
import numpy as np
import sounddevice as sd
import wavio
import whisper
import re
import threading
import queue
from collections import defaultdict
import torch
from pythonosc import udp_client
import os

vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
WAVE_OUTPUT_FILENAME = "audio_{0}.wav"
SILENCE_THRESHOLD = 1000
SILENCE_CHUNKS = 20

played_files = defaultdict(lambda: False)

model = whisper.load_model("small")

file_queue = queue.Queue()

# Set up the OSC client
ip = "127.0.0.1"
port = 9000
client = udp_client.SimpleUDPClient(ip, port)

def save_audio(filename, recording, samplerate=44100):
    data = np.array(recording * (2**15 - 1), dtype=np.int16)
    wavio.write(filename, data, samplerate)

def send_text_to_vrchat(text, send_immediately=True, play_sfx=True):
    # Send the text to the VRChat chatbox using the OSC address
    client.send_message("/chatbox/input", [text, send_immediately, play_sfx])

def speech_to_text(filename):
    data_file_path = os.path.join(os.path.dirname(__file__), filename)
    result = model.transcribe(data_file_path)
    text_result = result["text"]
    print(result["text"])

    if "exit app" in result["text"].lower():
        quit()
    if (re.sub(r'\W+', '', result["text"]).isalnum() and result["text"].isascii()):
        send_text_to_vrchat(result["text"])

def is_silent(data, threshold=SILENCE_THRESHOLD):
    return np.mean(np.abs(data)) < threshold

def process_audio_files():
    played_files = {}
    while True:
        if not file_queue.empty():
            filename = file_queue.get()
            played_files[filename] = False
            speech_to_text(filename)
            played_files[filename] = True

fileCount = 0

def record_audio():
    global fileCount
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Waiting for you to start talking...")

    while True:
        data = stream.read(CHUNK)
        data_np = np.frombuffer(data, dtype=np.int16)
        if not is_silent(data_np):
            print("Recording started!")
            break

    print("Recording...")
    frames = [data]
    silent_chunks = 0
    recorded_chunks = 0

    while True:
        data = stream.read(CHUNK)
        data_np = np.frombuffer(data, dtype=np.int16)
        frames.append(data)
        recorded_chunks += 1

        if is_silent(data_np):
            silent_chunks += 1
            if silent_chunks >= SILENCE_CHUNKS:
                break
        else:
            silent_chunks = 0

    stream.stop_stream()
    stream.close()
    p.terminate()


    if recorded_chunks > (RATE // CHUNK) // 3:  # Check if the recording duration is longer than 1 second
        print("Finished recording")
        # Find a played file to overwrite
        if fileCount > 4:
            fileCount = 0
            filename = WAVE_OUTPUT_FILENAME.format(fileCount)
        else:
            # If no played file is found, create a new one
            filename = WAVE_OUTPUT_FILENAME.format(fileCount)

        wf = wave.open(filename, "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))
        wf.close()


        # Check if the audio contains voice using Silero VAD
        wav = read_audio(filename, sampling_rate=16000)
        speech_timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=16000)

        if len(speech_timestamps) > 0:
            client.send_message("/chatbox/typing", [True])
            file_queue.put(filename)  # Add the file to the queue if there's voice detected
            fileCount = fileCount + 1
        else:
            print("No voice detected, not saving.")

    else:
        print("Recording too short, not saving.")
if __name__ == "__main__":
    processing_thread = threading.Thread(target=process_audio_files, daemon=True)
    processing_thread.start()

    while True:
        record_audio()

