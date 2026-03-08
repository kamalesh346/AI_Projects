import os
import json
from dotenv import load_dotenv
import re
load_dotenv()

from tts import speak
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from pdf2image import convert_from_path
import pytesseract

from openai import OpenAI
import httpx


# Create a more robust HTTP client with SSL configuration
http_client = httpx.Client(
    timeout=httpx.Timeout(60.0, read=30.0),
    verify=False,  # Disable SSL verification if needed
    limits=httpx.Limits(max_connections=10)
)

client = OpenAI(
    api_key=os.getenv("NAVIGATE_API_KEY"),
    base_url="https://apidev.navigatelabsai.com/",
    http_client=http_client
)


INDEX_DIR = "indexes"
CONV_DIR = "conversations"
MAX_SPEECH_CHARS = 300

os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(CONV_DIR, exist_ok=True)


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ---------------------------
# PDF LOADING WITH OCR FALLBACK
# ---------------------------
def load_pdf_with_ocr(pdf_path):

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text = "".join([doc.page_content.strip() for doc in documents])

    # If PDF has no readable text → use OCR
    if len(text) < 50:

        print("⚠️ No text found. Using OCR...")

        images = convert_from_path(pdf_path)

        ocr_docs = []

        for i, img in enumerate(images):

            extracted_text = pytesseract.image_to_string(img)

            ocr_docs.append(
                Document(
                    page_content=extracted_text,
                    metadata={"page": i}
                )
            )

        return ocr_docs

    return documents


# ---------------------------
# VECTOR INDEX MANAGEMENT
# ---------------------------
def load_or_create_index(pdf_path):

    pdf_name = Path(pdf_path).stem
    index_path = os.path.join(INDEX_DIR, pdf_name)

    faiss_file = os.path.join(index_path, "index.faiss")

    if os.path.exists(faiss_file):

        print("Loading existing index...")

        vectorstore = FAISS.load_local(
            index_path,
            embedding_model,
            allow_dangerous_deserialization=True
        )

    else:

        print("Creating new index...")

        documents = load_pdf_with_ocr(pdf_path)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=80
        )

        chunks = text_splitter.split_documents(documents)

        if len(chunks) == 0:
            raise ValueError("No text could be extracted from the PDF.")

        vectorstore = FAISS.from_documents(chunks, embedding_model)

        vectorstore.save_local(index_path)

        print("Index saved.")

    return vectorstore


# ---------------------------
# CONVERSATION MEMORY
# ---------------------------
def load_conversation(pdf_name):

    conv_path = os.path.join(CONV_DIR, f"{pdf_name}_chat.json")

    if os.path.exists(conv_path):

        with open(conv_path, "r") as f:
            return json.load(f)

    else:

        return [
            {
                "role": "system",
                "content": "Answer using only the provided document context."
            }
        ]


def save_conversation(pdf_name, memory):

    conv_path = os.path.join(CONV_DIR, f"{pdf_name}_chat.json")

    with open(conv_path, "w") as f:
        json.dump(memory, f, indent=2)

# ---------------------------
# CLEAN TEXT FOR TTS
# ---------------------------
def clean_text_for_tts(text):
    """Clean LLM output so it sounds natural in speech"""

    # remove markdown symbols
    text = re.sub(r"[#*_>`]", "", text)

    # remove bullet points
    text = re.sub(r"[-•]", "", text)

    # remove numbered lists like "1." "2."
    text = re.sub(r"\b\d+\.\s*", "", text)

    # remove urls
    text = re.sub(r"http\S+", "", text)

    # normalize spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# ---------------------------
# SPEECH POLICY
# ---------------------------
def should_skip_speech(text):
    """Detect responses that should not be spoken"""

    code_patterns = ["def ", "class ", "{", "}", "import "]

    for p in code_patterns:
        if p in text:
            return True

    # table detection
    if "|" in text:
        return True

    return False
# ---------------------------
# LLM RESPONSE GENERATION
# ---------------------------
def generate_answer_stream(question, context, memory):

    prompt = f"""
Answer ONLY using the provided context.
If answer is not found say:
'I could not find this information in the document.'

Context:
{context}
"""

    memory.append({"role": "user", "content": question})

    recent_memory = memory[-4:]

    stream = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=recent_memory + [{"role": "user", "content": prompt}],
        stream=True
    )

    full_text = ""
    sentence_buffer = ""
    spoken_chars = 0
    skip_speech = False
    for chunk in stream:

        delta = chunk.choices[0].delta

        if delta and delta.content:

            token = delta.content

            print(token, end="", flush=True)

            full_text += token
            if len(full_text) > 120 and not skip_speech:
                skip_speech = should_skip_speech(full_text)
            sentence_buffer += token

            # detect sentence ending
            # if any(p in token for p in [".", "!", "?"]):4
            # improved sentence detection
            if re.search(r"[.!?]\s$", sentence_buffer):

                sentence = sentence_buffer.strip()

                cleaned = clean_text_for_tts(sentence)

                if not skip_speech:

                    # always allow the first sentence
                    if spoken_chars == 0:
                        speak(cleaned)
                        spoken_chars += len(cleaned)

                    # allow more sentences if within limit
                    elif spoken_chars + len(cleaned) <= MAX_SPEECH_CHARS:
                        speak(cleaned)
                        spoken_chars += len(cleaned)

                    else:
                        skip_speech = True
                # wait to 2 seconds to show that TTS is streaming in real-time
                # time.sleep(2)
                sentence_buffer = ""

    # flush remaining text
    if sentence_buffer.strip():
        cleaned = clean_text_for_tts(sentence_buffer.strip())
        if not skip_speech:

            if spoken_chars == 0:
                speak(cleaned)

            elif spoken_chars + len(cleaned) <= MAX_SPEECH_CHARS:
                speak(cleaned)

    memory.append({"role": "assistant", "content": full_text})
    return full_text