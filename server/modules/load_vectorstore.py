import os
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY=(os.getenv("PINECONE_API_KEY") or "").strip()
PINECONE_ENV="us-east-1"
PINECONE_INDEX_NAME=os.getenv("PINECONE_INDEX_NAME", "medicalindex")

os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY

UPLOAD_DIR="./uploaded_docs"
os.makedirs(UPLOAD_DIR,exist_ok=True)


# initialize pinecone instance
pc=Pinecone(api_key=PINECONE_API_KEY)
spec=ServerlessSpec(cloud="aws",region=PINECONE_ENV)
existing_indexes=[i["name"] for i in pc.list_indexes()]


if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        # Must match the embedding vector size returned by
        # langchain_google_genai.GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        # In your case, Pinecone is returning: "Vector dimension 3072 does not match ... 768"
        dimension=3072,
        metric="dotproduct",
        spec=spec
    )
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)


index=pc.Index(PINECONE_INDEX_NAME)

# load,split,embed and upsert pdf docs content

def load_vectorstore(uploaded_files):
    expected_dimension = 3072
    # Safeguards for large PDFs: avoid huge embedding jobs that hit free-tier quotas.
    max_chunks_per_upload = int(os.getenv("MAX_CHUNKS_PER_UPLOAD", "800"))
    # Keep defaults aligned with the original implementation (500 / 50)
    # so re-uploads overwrite existing chunk IDs/metadata cleanly.
    chunk_size = int(os.getenv("EMBED_CHUNK_SIZE", "500"))
    chunk_overlap = int(os.getenv("EMBED_CHUNK_OVERLAP", "50"))
    embed_batch_size = int(os.getenv("EMBED_BATCH_SIZE", "64"))

    # Validate the Pinecone index dimension before upserting.
    # If it doesn't match the embedding dimension, Pinecone rejects the vectors.
    try:
        desc = pc.describe_index(PINECONE_INDEX_NAME)
        index_dimension = getattr(desc, "dimension", None) or desc.get("dimension")
    except Exception:
        index_dimension = None

    if index_dimension is not None and index_dimension != expected_dimension:
        raise RuntimeError(
            f"Pinecone index dimension mismatch: index={index_dimension}, expected={expected_dimension}. "
            f"Delete/recreate Pinecone index `{PINECONE_INDEX_NAME}` with dimension {expected_dimension}, "
            f"then re-upload PDFs."
        )

    embed_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    file_paths = []

    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())
        file_paths.append(str(save_path))

    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(documents)
        if len(chunks) > max_chunks_per_upload:
            chunks = chunks[:max_chunks_per_upload]

        texts = [chunk.page_content for chunk in chunks]
        # Pinecone metadata must include the actual chunk text,
        # because `/ask/` reconstructs `page_content` from `metadata["text"]`.
        metadatas = [{**(chunk.metadata or {}), "text": chunk.page_content} for chunk in chunks]
        ids = [f"{Path(file_path).stem}-{i}" for i in range(len(chunks))]

        print(f"🔍 Embedding {len(texts)} chunks...")
        embeddings = []
        # Embed in batches to reduce load and allow retry/backoff per batch.
        for i in tqdm(range(0, len(texts), embed_batch_size), desc="Embedding batches"):
            batch_texts = texts[i : i + embed_batch_size]
            batch_attempts = 0
            while True:
                batch_attempts += 1
                try:
                    embeddings.extend(embed_model.embed_documents(batch_texts))
                    break
                except Exception as e:
                    # Handle Gemini 429 "RESOURCE_EXHAUSTED" by waiting then retrying.
                    msg = str(e)
                    retry_delay_s = None
                    # Example in your error: "Please retry in 54.218010531s."
                    m = __import__("re").search(r"Please retry in ([0-9.]+)s", msg)
                    if m:
                        retry_delay_s = float(m.group(1))
                    if batch_attempts >= 3 or retry_delay_s is None:
                        raise
                    time.sleep(retry_delay_s + 2)

        print("📤 Uploading to Pinecone...")
        with tqdm(total=len(embeddings), desc="Upserting to Pinecone") as progress:
            index.upsert(vectors=zip(ids, embeddings, metadatas))
            progress.update(len(embeddings))

        print(f"✅ Upload complete for {file_path}")
