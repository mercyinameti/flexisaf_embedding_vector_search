# compare_search.py
# I run one question through vector search, keyword search, and hybrid search, and print the top results from each.

import json
import os
import re
import sys

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from rank_bm25 import BM25Okapi

load_dotenv()  # I read GEMINI_API_KEY from the .env file.
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.json"
TOP_K = 3  # I print the top 3 from each method.
RRF_CONSTANT = 60  # I use this standard constant when combining the two rankings.

question = " ".join(sys.argv[1:]) or "How do machines learn from examples?"

with open(CHUNKS_FILE, encoding="utf-8") as file:
    chunks = json.load(file)
index = faiss.read_index(INDEX_FILE)


def tokenize(text):
    # I lowercase the text and split it into words, dropping punctuation.
    return re.findall(r"[a-z0-9]+", text.lower())


# Vector ranking: every chunk, ordered by similarity to the question.
result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=question,
    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
)
query = np.array([result.embeddings[0].values], dtype="float32")
faiss.normalize_L2(query)
_, vector_rows = index.search(query, len(chunks))
vector_rank = [int(row) for row in vector_rows[0]]

# Keyword ranking: every chunk, ordered by BM25 score.
bm25 = BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])
keyword_scores = bm25.get_scores(tokenize(question))
keyword_rank = [int(row) for row in np.argsort(-keyword_scores)]

# Hybrid ranking: I add up each chunk's position in both lists, so a chunk near the top of either one rises.
hybrid_scores = {}
for ranking in (vector_rank, keyword_rank):
    for position, row in enumerate(ranking):
        hybrid_scores[row] = hybrid_scores.get(row, 0) + 1 / (RRF_CONSTANT + position + 1)
hybrid_rank = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)


def show(name, ranking):
    # I print the chunk IDs of the top results for one method.
    ids = [chunks[row]["chunk_id"] for row in ranking[:TOP_K]]
    print(f"{name:<8}: {', '.join(ids)}")


print(f"Question: {question}")
show("Vector", vector_rank)
show("Keyword", keyword_rank)
show("Hybrid", hybrid_rank)