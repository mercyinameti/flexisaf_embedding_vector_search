# build_index.py
# I turn every chunk into a vector, store the vectors in a FAISS index, and save the index to a file.

import json
import os

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai

from google.genai import types

load_dotenv()  # I read GEMINI_API_KEY from the .env file.
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

CHUNKS_FILE = "chunks.json"
INDEX_FILE = "index.faiss"

with open(CHUNKS_FILE, encoding="utf-8") as file:
    chunks = json.load(file)

texts = [chunk["text"] for chunk in chunks]  # I collect the text of every chunk.

# I send all the texts in one request, which is quicker than one request per chunk.
result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=texts,
    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),  # I mark these texts as documents to be searched.
)
vectors = np.array([item.values for item in result.embeddings], dtype="float32")

# I scale each vector to length 1, so a FAISS inner-product search gives cosine similarity.
faiss.normalize_L2(vectors)

index = faiss.IndexFlatIP(vectors.shape[1])  # I create an index that compares a query with every vector.
index.add(vectors)  # I store the vectors in the index.
faiss.write_index(index, INDEX_FILE)  # I save the index to a file, so I don't need to rebuild it.

print(f"Indexed {index.ntotal} chunks, each with {vectors.shape[1]} numbers, saved to {INDEX_FILE}")