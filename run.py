from openai import OpenAI
import httpx
import os
from dotenv import load_dotenv
from audio import record_audio, speech_to_text
from rag import generate_answer_stream, load_or_create_index, load_conversation, save_conversation
from pathlib import Path
import time
from tts import interrupt_speech, speak
import threading
import random

load_dotenv()

templates = [
    "I found relevant information for your query about {q}. According to the document, it appears on {p}.",
    "Looking at the document, information related to {q} appears on {p}.",
    "According to the document, the information about {q} is on {p}."
]
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

def retrieve_context(question, retriever):
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    pages = []
    for doc in docs:
        if "page" in doc.metadata:
            pages.append(doc.metadata["page"] + 1)

    pages = sorted(list(set(pages)))
    return context, pages

def speak_grounding(question, pages):

    rephrased_query = rephrase_query(question)

    if not pages:
        return

    template = random.choice(templates)

    if len(pages) == 1:
        page_text = f"page {pages[0]}"
    else:
        page_text = f"pages {', '.join(map(str, pages))}"

    message = template.format(q=rephrased_query, p=page_text)

    speak(message)

def rephrase_query(question):

    prompt = f"""
                Rephrase the user's query into a short clear phrase describing the information they want.

                User query:
                {question}

                Return only the rephrased phrase.
            """

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=30
    )

    return response.choices[0].message.content.strip()

def process_voice_input():
    audio_file = record_audio()

    if audio_file:
        question = speech_to_text(audio_file)
        print("You said:", question)
        return question

    return ""


def answer_question(question, retriever, memory):

    start = time.time()

    context, pages = retrieve_context(question, retriever)

    threading.Thread(
        target=speak_grounding,
        args=(question, pages),
        daemon=True
    ).start()
    answer = generate_answer_stream(question, context, memory)

    end = time.time()

    print(f"\n⏱️ Response time: {end-start:.2f}s")

    return answer


# -----------------------------

pdf_file = input("Enter PDF file path (inside data/): ")

vectorstore = load_or_create_index(pdf_file)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

pdf_name = Path(pdf_file).stem

memory = load_conversation(pdf_name)


print("\n🎙️ Voice PDF Assistant")
print("=" * 40)

mode = input("Choose mode: 'v' for voice or 't' for text: ").lower()

if mode not in ["v", "t"]:
    mode = "t"


while True:

    if mode == "v":

        question = process_voice_input()
        interrupt_words = ["stop", "wait", "hold on", "cancel"]

        if any(word in question for word in interrupt_words):
            print("Interrupt detected")
            interrupt_speech()
            continue

    else:

        question = input("Ask question: ")

    if not question.strip():
        continue


    if "exit" in question.lower():

        save_conversation(pdf_name, memory)

        print("Conversation saved.")

        break


    if "switch to voice" in question.lower():

        mode = "v"

        print("Switched to voice mode.")

        continue


    if "switch to text" in question.lower():

        mode = "t"

        print("Switched to text mode.")

        continue


    print(f"\nProcessing: {question}\n")

    answer = answer_question(question, retriever, memory)

    print(f"\nAssistant:\n{answer}\n")


save_conversation(pdf_name, memory)