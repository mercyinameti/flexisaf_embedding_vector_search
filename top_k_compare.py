# top_k_compare.py
# I run one question at three top-k settings and print the chunk IDs each setting returns.

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
KS = [1, 3, 5]  # I compare these three top-k settings.

question = " ".join(sys.argv[1:]) or "How do machines learn from examples?"

index = faiss.read_index(INDEX_FILE)
with open(CHUNKS_FILE, encoding="utf-8") as file:
    chunks = json.load(file)

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=question,
    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),  # I mark this text as a search query.
)
query = np.array([result.embeddings[0].values], dtype="float32")
faiss.normalize_L2(query)

scores, rows = index.search(query, max(KS))  # I search once, deep enough for the largest setting.
print(f"Question: {question}")
for k in KS:
    ids = [chunks[row]["chunk_id"] for row in rows[0][:k]]
    print(f"top-{k}: {', '.join(ids)}")