from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.embeddings.base import Embeddings
import os
import re

# Configuration constants
CHUNK_OUTPUT_ROOT = "./subject_chunks"  # Must match semantic_chunker.py
VECTOR_STORE_ROOT = "./vector_stores"


def create_vector_stores():
    embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
    
    # Create output directory if not exists
    os.makedirs(VECTOR_STORE_ROOT, exist_ok=True)
    
    # Verify chunk directory exists
    if not os.path.exists(CHUNK_OUTPUT_ROOT):
        raise FileNotFoundError(
            f"Chunk directory {CHUNK_OUTPUT_ROOT} not found. "
            "Run semantic_chunker.py first."
        )

    # for subject_name in os.listdir(CHUNK_OUTPUT_ROOT):
        
    subject_name = "Mobile_Computing"

    chunk_file = os.path.join(CHUNK_OUTPUT_ROOT, subject_name, "combined_chunks.txt")

    if not os.path.exists(chunk_file):
        print(f"Skipping {subject_name} - no combined chunks found")
        # continue

    try:
        with open(chunk_file, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except UnicodeDecodeError as e:
        print(f"Error reading {chunk_file}: {e}")
        # continue

    # Use lookahead-based regex to correctly split chunks by starting markers only
    pattern = r"(=== Page \d+ ===(?:.*?))(?=(\n=== Page \d+ ===|\Z))"
    matches = re.findall(pattern, content, re.DOTALL)
    chunks = [match[0].strip() for match in matches if match[0].strip()]

    if not chunks:
        print(f"Skipping {subject_name} - no valid chunks found")
        # continue

    # Create and save vector store
    vs_path = os.path.join(VECTOR_STORE_ROOT, subject_name)
    os.makedirs(vs_path, exist_ok=True)

    try:
        vector_store = FAISS.from_texts(chunks, embedding_model)
        vector_store.save_local(vs_path)
        print(f"✅ Created vector store for {subject_name} with {len(chunks)} chunks")
    except Exception as e:
        print(f"❌ Failed to create vector store for {subject_name}: {str(e)}")

if __name__ == "__main__":
    create_vector_stores()
