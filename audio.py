import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from openai import OpenAI
import httpx

# Create a more robust HTTP client with SSL configuration
http_client = httpx.Client(
    timeout=httpx.Timeout(60.0, read=30.0),
    verify=False,  # Disable SSL verification if needed
    limits=httpx.Limits(max_connections=10)
)

client = OpenAI(
    api_key="sk-5MOyzg2xQQiJT1uYcZ9Tzg",
    base_url="https://apidev.navigatelabsai.com/",
    http_client=http_client
)

SAMPLE_RATE = 16000
CHUNK_SIZE = 512   # 1024

# speech detection parameters
VOICE_THRESHOLD = 0.2
SILENCE_SECONDS = 1.5   # 2
MIN_SPEECH_SECONDS = 0.6


def record_audio(filename="question.wav"):

    print("🎤 Listening...")

    recording = []
    speaking = False
    silence_counter = 0
    speech_frames = 0

    silence_limit = int((SILENCE_SECONDS * SAMPLE_RATE) / CHUNK_SIZE)
    min_speech_frames = int((MIN_SPEECH_SECONDS * SAMPLE_RATE) / CHUNK_SIZE)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1) as stream:

        while True:

            audio_chunk, _ = stream.read(CHUNK_SIZE)

            volume = np.linalg.norm(audio_chunk)
            print("Volume:", volume) 

            # detect speech start
            if volume > VOICE_THRESHOLD:

                if not speaking:
                    print("🗣 Speech detected...")

                speaking = True
                silence_counter = 0
                speech_frames += 1
                recording.append(audio_chunk)

            else:

                if speaking:
                    silence_counter += 1
                    recording.append(audio_chunk)

                    # speech ended
                    if silence_counter > silence_limit:

                        # ensure minimum speech duration
                        if speech_frames >= min_speech_frames:
                            print("✅ Speech ended")
                            break
                        else:
                            # reset if speech was too short
                            speaking = False
                            silence_counter = 0
                            speech_frames = 0
                            recording = []

                # if user hasn't spoken yet just keep listening
                continue

    if len(recording) == 0:
        return None

    audio = np.concatenate(recording)

    write(filename, SAMPLE_RATE, audio)

    return filename


def speech_to_text(audio_path):

    if audio_path is None:
        return ""

    with open(audio_path, "rb") as audio_file:
        
        # Add retry logic and error handling for speech-to-text
        max_retries = 3
        for attempt in range(max_retries):
            try:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
                return transcription.text.strip().lower()
            except Exception as e:
                print(f"Speech-to-text attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    print("Sorry, speech recognition is not available right now.")
                    return ""  # Return empty string if all retries failed
                import time
                time.sleep(2)  # Wait 2 seconds before retry