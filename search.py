# search.py
# I turn a question into a vector, find the closest chunks in the index, and print them.

import json
import os
import sys

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai

from google.genai import types

load_dotenv()  # I read GEMINI_API_KEY from the .env file.
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.json"
TOP_K = 3  # I return the 3 closest chunks.

# I take the question from the command line, or use a default question.
question = " ".join(sys.argv[1:]) or "What is machine learning?"

index = faiss.read_index(INDEX_FILE)
with open(CHUNKS_FILE, encoding="utf-8") as file:
    chunks = json.load(file)

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=question,
    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),  # I mark this text as a search query.
)
query = np.array([result.embeddings[0].values], dtype="float32")
faiss.normalize_L2(query)  # I scale the question vector the same way I scaled the chunk vectors.

scores, rows = index.search(query, TOP_K)
for rank, (score, row) in enumerate(zip(scores[0], rows[0]), start=1):
    chunk = chunks[row]
    print(f"\n{rank}. {chunk['chunk_id']} ({chunk['title']}) score {score:.3f}")
    print(chunk["text"])