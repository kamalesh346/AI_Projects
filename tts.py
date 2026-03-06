import io
import sounddevice as sd
import soundfile as sf
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

def speak(text):
    """Speak text using TTS with proper threading support"""
    print(f"🔊 Starting TTS for: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    
    try:
        # Add retry logic and error handling for TTS
        max_retries = 3
        response = None
        
        for attempt in range(max_retries):
            try:
                print(f"🎵 TTS attempt {attempt + 1}...")
                response = client.audio.speech.create(
                    model="gpt-4o-mini-tts",
                    voice="alloy",
                    input=text
                )
                print("✅ TTS request successful!")
                break
            except Exception as e:
                print(f"❌ TTS attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    print("🚨 Sorry, text-to-speech is not available right now.")
                    # Fallback: Use system TTS or just print
                    print("🔄 Trying fallback TTS methods...")
                    
                    # Try Windows SAPI (if on Windows)
                    if try_windows_tts(text):
                        return
                    
                    # Try pyttsx3 (cross-platform)
                    if try_pyttsx3_tts(text):
                        return
                        
                    # Final fallback: just display text
                    print("📢 [AUDIO WOULD SAY]:")
                    print(f"🗣️  {text}")
                    print("📢 [END AUDIO]")
                    return
                import time
                time.sleep(2)  # Wait 2 seconds before retry

        if response is None:
            print("❌ No TTS response received")
            return

        # Convert response bytes to in-memory audio buffer
        print("🎧 Processing audio...")
        audio_buffer = io.BytesIO(response.content)

        # Decode and play audio
        data, samplerate = sf.read(audio_buffer)    
        print(f"▶️  Playing audio ({len(data)} samples at {samplerate}Hz)...")
        
        # Play audio and wait for completion
        sd.play(data, samplerate)
        sd.wait()  # This is crucial - wait for playback to complete
        
        print("✅ TTS playback completed")
        
    except Exception as e:
        print(f"❌ TTS Error: {e}")
        print("📢 [AUDIO WOULD SAY]:")
        print(f"🗣️  {text}")
        print("📢 [END AUDIO]")


def try_windows_tts(text):
    """Try Windows SAPI TTS"""
    try:
        import win32com.client
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        speaker.Speak(text)
        print("✅ Used Windows SAPI TTS")
        return True
    except Exception:
        return False


def try_pyttsx3_tts(text):
    """Try pyttsx3 TTS"""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        print("✅ Used pyttsx3 TTS")
        return True
    except Exception:
        return False


def test_audio_output():
    """Test if audio output is working"""
    print("🔊 Testing audio output...")
    try:
        import sounddevice as sd
        import numpy as np
        
        # Generate test tone
        duration = 0.5
        freq = 440
        samplerate = 44100
        
        t = np.linspace(0, duration, int(samplerate * duration))
        tone = 0.3 * np.sin(2 * np.pi * freq * t)
        
        sd.play(tone, samplerate)
        sd.wait()
        print("✅ Audio system is working!")
        return True
    except Exception as e:
        print(f"❌ Audio system test failed: {e}")
        return False