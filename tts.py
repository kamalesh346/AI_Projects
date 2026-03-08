import io
import queue
import threading
import sounddevice as sd
import soundfile as sf
from openai import OpenAI
import httpx
import os
from dotenv import load_dotenv
import random

load_dotenv()

http_client = httpx.Client(
    timeout=httpx.Timeout(60.0, read=30.0),
    verify=False,
    limits=httpx.Limits(max_connections=10)
)

client = OpenAI(
    api_key=os.getenv("NAVIGATE_API_KEY"),
    base_url="https://apidev.navigatelabsai.com/",
    http_client=http_client
)


# Audio queue
speech_queue = queue.Queue()


def audio_worker():
    """Continuously plays audio from queue"""

    while True:

        text = speech_queue.get()

        if text is None:
            break

        try:
            print("Speaking:", text)  # debug print
            response = client.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice="alloy",
                input=text
            )

            audio_buffer = io.BytesIO(response.content)

            data, samplerate = sf.read(audio_buffer)

            sd.play(data, samplerate)
            sd.wait()

        except Exception as e:
            print("TTS error:", e)

        speech_queue.task_done()


# Start background audio thread
threading.Thread(target=audio_worker, daemon=True).start()

def interrupt_speech():

    # stop currently playing audio
    sd.stop()
    
    # clear queued speech
    try:
        while True:
            speech_queue.get_nowait()
            speech_queue.task_done()
    except queue.Empty:
        pass

    responses = [
        "Okay, stopping. What would you like to ask?",
        "Sure, go ahead.",
        "Alright, I'm listening.",
        "Got it. What's now?"
    ]

    # acknowledge interruption
    speak(random.choice(responses))

def speak(text):
    """Add speech to playback queue"""
    speech_queue.put(text)