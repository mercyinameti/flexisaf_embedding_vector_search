# filter_search.py
# I search only the chunks in the category I ask for, then print the closest ones.

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
TOP_K = 3  # I print at most 3 matching chunks.

# I read the category from the first argument, and the question from the rest.
category = sys.argv[1]
question = " ".join(sys.argv[2:])

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

# I rank all chunks by similarity, then keep only the ones in the requested category.
scores, rows = index.search(query, index.ntotal)
matches = [(score, row) for score, row in zip(scores[0], rows[0]) if chunks[row]["category"] == category]

if not matches:
    print(f"No chunks found in category '{category}'.")
else:
    for rank, (score, row) in enumerate(matches[:TOP_K], start=1):
        chunk = chunks[row]
        print(f"\n{rank}. {chunk['chunk_id']} ({chunk['title']}) score {score:.3f}")
        print(chunk["text"])