# diagnose.py
# I show the top results for one evaluation question, and where each expected chunk ranks.

import csv
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
QUESTIONS_FILE = "evaluation_questions.csv"
RRF_CONSTANT = 60
QUESTION_ID = sys.argv[1] if len(sys.argv) > 1 else "Q04"  # I default to question 4.


def tokenize(text):
    # I lowercase the text and split it into words, dropping punctuation.
    return re.findall(r"[a-z0-9]+", text.lower())


with open(CHUNKS_FILE, encoding="utf-8") as file:
    chunks = json.load(file)
index = faiss.read_index(INDEX_FILE)
bm25 = BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])

with open(QUESTIONS_FILE, encoding="utf-8", newline="") as file:
    question = next(q for q in csv.DictReader(file) if q["question_id"] == QUESTION_ID)
expected = question["expected_chunk_ids"].split(";")

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=question["question"],
    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
)
query = np.array([result.embeddings[0].values], dtype="float32")
faiss.normalize_L2(query)
_, vector_rows = index.search(query, len(chunks))
vector_rank = [int(row) for row in vector_rows[0]]

keyword_scores = bm25.get_scores(tokenize(question["question"]))
keyword_rank = [int(row) for row in np.argsort(-keyword_scores)]

hybrid_scores = {}
for ranking in (vector_rank, keyword_rank):
    for place, row in enumerate(ranking):
        hybrid_scores[row] = hybrid_scores.get(row, 0) + 1 / (RRF_CONSTANT + place + 1)
hybrid_rank = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)

chunk_ids = [chunk["chunk_id"] for chunk in chunks]
print(f"{QUESTION_ID}: {question['question']}")
print(f"Expected: {', '.join(expected)}")

for name, ranking in [("Vector", vector_rank), ("Keyword", keyword_rank), ("Hybrid", hybrid_rank)]:
    ranked_ids = [chunk_ids[row] for row in ranking]
    positions = {chunk_id: ranked_ids.index(chunk_id) + 1 for chunk_id in expected}
    print(f"\n{name} top 5: {', '.join(ranked_ids[:5])}")
    print(f"{name} position of expected chunks: {positions}")

print(f"\nVector top-1 text:\n{chunks[vector_rank[0]]['text']}")