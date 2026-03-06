import os
import json
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
    api_key="sk-5MOyzg2xQQiJT1uYcZ9Tzg",
    base_url="https://apidev.navigatelabsai.com/",
    http_client=http_client
)


INDEX_DIR = "indexes"
CONV_DIR = "conversations"

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
# LLM RESPONSE GENERATION
# ---------------------------
def generate_answer(question, context, memory):

    prompt = f"""
Answer ONLY using the provided context.
If the answer is not present, say:
"I could not find this information in the document."

Context:
{context}
"""

    memory.append({"role": "user", "content": question})

    # keep last few turns only (reduces latency)
    recent_memory = memory[-4:]

    # Add retry logic and error handling
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=recent_memory + [{"role": "user", "content": prompt}]
            )
            answer = response.choices[0].message.content
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                return "Sorry, I'm having trouble connecting to the AI service right now. Please check your internet connection and try again later."
            import time
            time.sleep(2)  # Wait 2 seconds before retry

    memory.append({"role": "assistant", "content": answer})

    return answer