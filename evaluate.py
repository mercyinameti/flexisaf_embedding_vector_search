# evaluate.py
# I run every evaluation question through vector, keyword, and hybrid search, then score each method.

import csv
import json
import os
import re

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
RESULTS_FILE = "evaluation_results.csv"
KS = [1, 3, 5]  # I check the top 1, top 3, and top 5 results.
RRF_CONSTANT = 60  # I use this constant when combining the vector and keyword rankings.
METHODS = ["Vector", "Keyword", "Hybrid"]


def tokenize(text):
    # I lowercase the text and split it into words, dropping punctuation.
    return re.findall(r"[a-z0-9]+", text.lower())


with open(CHUNKS_FILE, encoding="utf-8") as file:
    chunks = json.load(file)
index = faiss.read_index(INDEX_FILE)
bm25 = BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])

with open(QUESTIONS_FILE, encoding="utf-8", newline="") as file:
    questions = list(csv.DictReader(file))

# I embed all the questions in one request.
result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=[q["question"] for q in questions],
    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
)
query_vectors = np.array([item.values for item in result.embeddings], dtype="float32")
faiss.normalize_L2(query_vectors)
_, vector_rows = index.search(query_vectors, len(chunks))


def rankings_for(position, question):
    # I build the three rankings for one question: every chunk, best first.
    vector_rank = [int(row) for row in vector_rows[position]]
    keyword_scores = bm25.get_scores(tokenize(question))
    keyword_rank = [int(row) for row in np.argsort(-keyword_scores)]

    hybrid_scores = {}
    for ranking in (vector_rank, keyword_rank):
        for place, row in enumerate(ranking):
            hybrid_scores[row] = hybrid_scores.get(row, 0) + 1 / (RRF_CONSTANT + place + 1)
    hybrid_rank = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)

    return {"Vector": vector_rank, "Keyword": keyword_rank, "Hybrid": hybrid_rank}


def is_hit(ranking, expected_ids, k):
    # I count a hit when at least one correct chunk is in the top k.
    top_ids = {chunks[row]["chunk_id"] for row in ranking[:k]}
    return bool(top_ids & expected_ids)


totals = {(method, k): 0 for method in METHODS for k in KS}
results = []
for position, q in enumerate(questions):
    expected_ids = set(q["expected_chunk_ids"].split(";"))
    rankings = rankings_for(position, q["question"])
    row = {"question_id": q["question_id"], "question_type": q["question_type"]}
    for method in METHODS:
        for k in KS:
            hit = is_hit(rankings[method], expected_ids, k)
            row[f"{method}_top{k}"] = "yes" if hit else "no"
            totals[(method, k)] += hit
    results.append(row)

with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

print(f"Hits out of {len(questions)} questions, by the top k results checked:")
for method in METHODS:
    scores = "   ".join(f"top-{k}: {totals[(method, k)]}/{len(questions)}" for k in KS)
    print(f"{method:<8} {scores}")
print(f"Per-question results saved to {RESULTS_FILE}")