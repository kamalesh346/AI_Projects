from audio import record_audio, speech_to_text
from rag import load_or_create_index, generate_answer, load_conversation, save_conversation
from tts import speak
from pathlib import Path
import threading
import time


def parallel_process_question(question, retriever, memory):
    """
    Process question with parallel RAG retrieval and LLM preparation
    """
    context_result = {}
    answer_result = {}
    
    def retrieve_context():
        """Retrieve relevant documents from vector store"""
        try:
            start_time = time.time()
            docs = retriever.invoke(question)
            context = "\n\n".join([doc.page_content for doc in docs])
            context_result['context'] = context
            context_result['time'] = time.time() - start_time
            print(f" Context retrieved in {context_result['time']:.2f}s")
        except Exception as e:
            context_result['context'] = ""
            context_result['error'] = str(e)
    
    def prepare_llm_request():
        """Prepare LLM request parameters (could include prompt engineering, memory processing)"""
        try:
            start_time = time.time()
            # Pre-process memory and prepare prompt template
            recent_memory = memory[-4:] if memory else []
            answer_result['recent_memory'] = recent_memory
            answer_result['prep_time'] = time.time() - start_time
            print(f" LLM prep completed in {answer_result['prep_time']:.2f}s")
        except Exception as e:
            answer_result['recent_memory'] = []
            answer_result['error'] = str(e)
    
    # Start both operations in parallel
    print(" Starting parallel processing...")
    
    context_thread = threading.Thread(target=retrieve_context)
    llm_prep_thread = threading.Thread(target=prepare_llm_request)
    
    parallel_start = time.time()
    
    context_thread.start()
    llm_prep_thread.start()
    
    # Wait for both to complete
    context_thread.join()
    llm_prep_thread.join()
    
    parallel_time = time.time() - parallel_start
    print(f" Parallel processing completed in {parallel_time:.2f}s")
    
    # Now generate the answer with the prepared context and memory
    if 'context' in context_result and 'recent_memory' in answer_result:
        llm_start = time.time()
        answer = generate_answer(question, context_result['context'], memory)
        llm_time = time.time() - llm_start
        print(f"LLM response generated in {llm_time:.2f}s")
        return answer
    else:
        return "Sorry, I encountered an error processing your request."


def process_voice_input_optimized():
    """
    Optimized voice input processing with parallel operations
    """
    print(" Recording audio...")
    record_start = time.time()
    audio_file = record_audio()
    record_time = time.time() - record_start
    print(f" Audio recorded in {record_time:.2f}s")
    
    if audio_file:
        print(" Converting speech to text...")
        stt_start = time.time()
        question = speech_to_text(audio_file)
        stt_time = time.time() - stt_start
        print(f" Speech-to-text completed in {stt_time:.2f}s")
        print("You said:", question)
        return question
    return ""


pdf_file = input("Enter PDF file path (inside data/): ")

vectorstore = load_or_create_index(pdf_file)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

pdf_name = Path(pdf_file).stem
memory = load_conversation(pdf_name)


# Ask mode only once
print("\n🎙️ VOICE PDF ASSISTANT")
print("=" * 40)
print("🎤 Voice Mode: Speak your questions, hear responses")
print("⌨️  Text Mode: Type questions, read responses")
print("\n⚙️  Commands: 'exit', 'switch to voice', 'switch to text'")
mode = input("\n📦 Choose mode: 'v' for voice or 't' for text: ").lower()

if mode == "v":
    print("\n🎤 VOICE MODE ACTIVATED")
    print("📶 Please ensure your microphone and speakers are working")
    print("🎵 You will hear responses through your speakers")
    print("⏸️  Speak clearly and wait for the beep")
elif mode == "t":
    print("\n⌨️  TEXT MODE ACTIVATED")
    print("📝 Type your questions and press Enter")
else:
    print("\n⚠️  Invalid mode, defaulting to text mode")
    mode = "t"


while True:
    total_start = time.time()
    
    if mode == "v":
        question = process_voice_input_optimized()

    elif mode == "t":
        question = input("Ask question: ")

    else:
        print("Invalid mode. Switching to text.")
        mode = "t"
        continue

    if not question.strip():
        print("Didn't catch that. Try again.")
        continue

    # Exit command
    if "exit" in question.lower():
        save_conversation(pdf_name, memory)
        print("Conversation saved.")
        break

    # Mode switch commands
    if "switch to voice" in question.lower():
        mode = "v"
        print("Switched to voice mode.")
        continue

    if "switch to text" in question.lower():
        mode = "t"
        print("Switched to text mode.")
        continue

    # Use optimized parallel processing
    print(f"\n Processing: '{question}'")
    answer = parallel_process_question(question, retriever, memory)

    print(f"\n🤖 Assistant:\n{answer}")

    # Handle TTS properly for voice mode
    tts_thread = None
    if mode == "v":
        print("\n🎵 Starting text-to-speech...")
        print("📶 Make sure your speakers/headphones are on!")
        
        tts_thread = threading.Thread(target=speak, args=(answer,))
        tts_thread.start()
        
        # Give TTS a moment to start
        time.sleep(0.5)
        
        # For better UX, always wait for TTS to complete in voice mode
        # This ensures users hear the full response before continuing
        print("⏳ Playing audio response... (waiting for completion)")
        tts_thread.join()
        print("✅ Audio playback finished!")
        print("🎤 Ready for your next question...")
    
    total_time = time.time() - total_start
    print(f"\n⏱️ Total response time: {total_time:.2f}s")
    print(f"{'='*60}")

    save_conversation(pdf_name, memory)