import os
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables from .env
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medicalindex")
PINECONE_ENV = "us-east-1"  # You can change the region if needed

if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY not found in environment variables!")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Upload directory for PDFs
UPLOAD_DIR = "./uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)
existing_indexes = [i["name"] for i in pc.list_indexes()]

def _get_index_dimension(name: str) -> int | None:
    """Return the dimension for an existing index, or None if not found."""
    try:
        desc = pc.describe_index(name)
        # Newer pinecone client returns an object with .dimension, older may be dict
        if hasattr(desc, "dimension") and desc.dimension is not None:
            return int(desc.dimension)
        # Fallback if dict-like
        if isinstance(desc, dict):
            dim = desc.get("dimension") or (desc.get("spec", {}) or {}).get("dimension")
            return int(dim) if dim else None
    except Exception:
        return None
    return None

def _pick_model_for_dim(dim: int) -> tuple[str, dict]:
    """Map dimension to a suitable HuggingFace model and encode kwargs."""
    # Default to normalizing for dotproduct
    encode_kwargs = {"normalize_embeddings": True}
    if dim == 384:
        return "sentence-transformers/all-MiniLM-L6-v2", encode_kwargs
    if dim == 768:
        return "sentence-transformers/all-mpnet-base-v2", encode_kwargs
    if dim == 1024:
        # e5-large-v2 outputs 1024 dims
        return "intfloat/e5-large-v2", encode_kwargs
    raise ValueError(
        f"Unsupported Pinecone index dimension {dim}. Update mapping in load_vectorstore.py or recreate index."
    )

# Create index if it doesn't exist
index_dimension: int | None = None
if PINECONE_INDEX_NAME not in existing_indexes:
    # If creating fresh, default to 384 (MiniLM) unless overridden via env
    default_dim = int(os.getenv("EMBEDDING_DIM", 384))
    print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}' with dim {default_dim}...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=default_dim,
        metric="dotproduct",
        spec=spec,
    )
    # Wait for index to be ready
    while True:
        desc = pc.describe_index(PINECONE_INDEX_NAME)
        ready = False
        if hasattr(desc, "status"):
            # status may be dict-like
            st = desc.status if isinstance(desc.status, dict) else getattr(desc.status, "ready", None)
            ready = st["ready"] if isinstance(st, dict) else bool(st)
        elif isinstance(desc, dict):
            ready = bool(((desc.get("status") or {})).get("ready"))
        if ready:
            break
        time.sleep(1)
    index_dimension = default_dim
    print("✅ Pinecone index ready!")
else:
    # If it already exists, read its dimension to choose a matching embedding model
    index_dimension = _get_index_dimension(PINECONE_INDEX_NAME)
    if not index_dimension:
        # As a safe fallback, keep going and let Pinecone validate later
        index_dimension = 384

index = pc.Index(PINECONE_INDEX_NAME)

# Function to load, split, embed, and upsert PDFs
def load_vectorstore(uploaded_files):
    # Select embedding model that matches the Pinecone index dimension
    model_name, encode_kwargs = _pick_model_for_dim(index_dimension)
    embed_model = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs=encode_kwargs,
    )

    file_paths = []

    # Save uploaded files
    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())
        file_paths.append(str(save_path))

    # Process each file
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        texts = [chunk.page_content for chunk in chunks]
        # Include the text in metadata so query can reconstruct docs
        metadatas = []
        for ch in chunks:
            md = dict(ch.metadata) if isinstance(ch.metadata, dict) else {}
            md["text"] = ch.page_content
            metadatas.append(md)
        ids = [f"{Path(file_path).stem}-{i}" for i in range(len(chunks))]

        print(f"🔍 Embedding {len(texts)} chunks using HuggingFace...")
        embeddings = embed_model.embed_documents(texts)

        print("📤 Uploading to Pinecone...")
        vectors = [(ids[i], embeddings[i], metadatas[i]) for i in range(len(embeddings))]
        with tqdm(total=len(vectors), desc="Upserting to Pinecone") as progress:
            index.upsert(vectors=vectors)
            progress.update(len(vectors))

        print(f"✅ Upload complete for {file_path}")
