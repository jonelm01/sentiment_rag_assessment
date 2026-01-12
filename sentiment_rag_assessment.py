# Step 1: Install Dependencies
# Using %pip ensures packages are installed in the current Jupyter kernel
%pip install numpy boto3 chromadb langchain langchain-community langchain-aws langchain-text-splitters


# Step 2: Configuration & Variables
import os
import chromadb
from chromadb.config import Settings

# --- AWS Configuration ---
# PLEASE REPLACE WITH YOUR ACTUAL CREDENTIALS
AWS_ACCESS_KEY_ID = "XXX"
AWS_SECRET_ACCESS_KEY = "XXX"
AWS_REGION = "us-west-2"

# --- Bedrock Model Configuration ---
# Using a stable Claude 3 Sonnet ID which is widely available in us-west-2
BEDROCK_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"

# --- ChromaDB Cloud Configuration ---
# Sign up at https://trychroma.com to get your API Token
CHROMA_API_KEY = "ck-XXX"
CHROMA_TENANT = "default_tenant"  # Usually 'default_tenant' for most users
CHROMA_DATABASE = "db_1" # Usually 'default_database'
CHROMA_COLLECTION_NAME = "rag_collection"

# Apply Environment Variables for Boto3
os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
os.environ["AWS_DEFAULT_REGION"] = AWS_REGION

print("Configuration Loaded.")

# ======================================
# STEP 1–3: Initialization 
# ======================================
import boto3
import chromadb

print("1. Initializing Boto3 Session...")
session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)
bedrock_client = session.client("bedrock-runtime")
print("Bedrock Client Initialized successfully.")

print("\n 2. Initializing ChromaDB Client...")
chroma_client = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
print(f"Connected to Chroma. Collection '{CHROMA_COLLECTION_NAME}' ready.")
print(f"Collection Count: {collection.count()}")


# ======================================
# STEP 4: DIRECTORIES 
# ======================================
import os

SOURCE_DIR = "files"
CHUNKED_DIR = os.path.join(SOURCE_DIR, "chunked")

if not os.path.exists(CHUNKED_DIR):
    os.makedirs(CHUNKED_DIR)
    print(f"Created directory: {CHUNKED_DIR}")
else:
    print(f"Directory exists: {CHUNKED_DIR}")

source_files = [
    f for f in os.listdir(SOURCE_DIR)
    if os.path.isfile(os.path.join(SOURCE_DIR, f)) and not f.startswith('.')
]
print(f"Found {len(source_files)} files: {source_files[:5]}")


# ======================================
# STEP 5: SEMANTIC CHUNKING
# ======================================
from sentence_transformers import SentenceTransformer, util
import nltk

semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

import re

def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text)

def semantic_chunk(text, max_tokens=350, sim_threshold=0.63):
    sentences = split_sentences(text)
    chunks = []
    current = []

    for s in sentences:
        if not current:
            current.append(s)
            continue
        
        emb1 = semantic_model.encode(" ".join(current), convert_to_numpy=False, normalize_embeddings=True)
        emb2 = semantic_model.encode(s, convert_to_numpy=False, normalize_embeddings=True)
        sim = util.cos_sim(emb1, emb2).item()

        if sim > sim_threshold and len(" ".join(current + [s]).split()) < max_tokens:
            current.append(s)
        else:
            chunks.append(" ".join(current))
            current = [s]

    if current:
        chunks.append(" ".join(current))

    return chunks



total_chunks_processed = 0
print("Starting semantic chunking...\n")

for file_name in source_files:
    file_path = os.path.join(SOURCE_DIR, file_name)

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    chunks = semantic_chunk(text)

    base_name = os.path.splitext(file_name)[0]

    for i, chunk in enumerate(chunks):
        fname = f"ch{i+1}-{base_name}-len{len(chunk)}.txt"
        with open(os.path.join(CHUNKED_DIR, fname), 'w', encoding='utf-8') as out:
            out.write(chunk)

    print(f" {file_name}: {len(chunks)} semantic chunks")
    total_chunks_processed += len(chunks)

print(f"\n Total Semantic Chunks: {total_chunks_processed}")


# ======================================
# STEP 7: LOAD CHUNKS → PREP FOR EMBEDDING
# ======================================
import uuid

chunked_files = [f for f in os.listdir(CHUNKED_DIR) if f.endswith('.txt')]
documents, metadatas, ids = [], [], []

print(f"Found {len(chunked_files)} chunks to embed.")

for file_name in chunked_files:
    file_path = os.path.join(CHUNKED_DIR, file_name)

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    name_no_ext = os.path.splitext(file_name)[0]
    parts = name_no_ext.split('-')

    try:
        chunk_part = int(parts[0].replace('ch',''))
        size = int(parts[-1].replace('len',''))
        original_filename = "-".join(parts[1:-1])
        meta = {"source": file_name, "file_name": original_filename, "chunk": chunk_part, "size": size}
    except:
        meta = {"source": file_name}

    documents.append(content)
    metadatas.append(meta)
    ids.append(str(uuid.uuid4()))

print(f"Prepared {len(documents)} docs for embedding.")


# ======================================
# STEP 8: UPSERT INTO CHROMA
# ======================================
BATCH_SIZE = 100
print("Upserting documents into Chroma...")

for i in range(0, len(documents), BATCH_SIZE):
    collection.add(
        documents=documents[i:i+BATCH_SIZE],
        metadatas=metadatas[i:i+BATCH_SIZE],
        ids=ids[i:i+BATCH_SIZE]
    )
    print(f"Batch {i} → {min(i+BATCH_SIZE, len(documents))}")

print("Chroma upsert complete.")
print(f"Final Count: {collection.count()}")

from langchain_aws import ChatBedrock


llm = ChatBedrock(
    model_id=BEDROCK_MODEL_ID,
    client=bedrock_client,
    model_kwargs={"max_tokens": 400, "temperature": 0}
)
# ======================================
# STEP 10: RETRIEVAL 
# ======================================
def retrieve_semantic(query, n_results=6, threshold=1.2):
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    docs = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        if dist <= threshold:
            docs.append({
                "text": text,
                "source": meta.get("source","unknown"),
                "dist": round(dist,3)
            })

    return docs


# ======================================
# STEP 11: FINAL RAG ANSWER GENERATION
# ======================================
def generate_answer(query):
    docs = retrieve_semantic(query)

    if not docs:
        return "No relevant info in the knowledge base."

    context = "\n\n---\n\n".join(f"[{d['source']}] (dist={d['dist']})\n{d['text']}" for d in docs)

    prompt = f"""
Use context to answer the question.
Cite sources inline like [filename].
If unknown, say 'I don't know'.

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)
    return response.content


# ======================================
# STEP 12: TEST
# ======================================
q = "What is the policy regarding drug usage?"
print("Query:", q)
print("Answer:", generate_answer(q))
