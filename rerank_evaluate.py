# rerank_evaluate.py
# I take vector search's top 5 for each question, ask a Gemini model to reorder them, and score the new order.

import csv
import json
import os
import re

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()  # I read GEMINI_API_KEY from the .env file.
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.json"
QUESTIONS_FILE = "evaluation_questions.csv"
RESULTS_FILE = "rerank_results.csv"
CANDIDATES = 5  # I re-rank only the top 5 vector results.
KS = [1, 3, 5]
LETTERS = ["A", "B", "C", "D", "E"]
EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash-lite"

with open(CHUNKS_FILE, encoding="utf-8") as file:
    chunks = json.load(file)
index = faiss.read_index(INDEX_FILE)

with open(QUESTIONS_FILE, encoding="utf-8", newline="") as file:
    questions = list(csv.DictReader(file))

# I embed all the questions in one request, then find each question's top results.
result = client.models.embed_content(
    model=EMBED_MODEL,
    contents=[q["question"] for q in questions],
    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
)
query_vectors = np.array([item.values for item in result.embeddings], dtype="float32")
faiss.normalize_L2(query_vectors)
_, vector_rows = index.search(query_vectors, len(chunks))


def rerank(question, candidate_rows):
    # I ask the model to order the candidate passages by how well each one answers the question.
    passages = "\n\n".join(
        f"[{LETTERS[i]}] {chunks[row]['text']}" for i, row in enumerate(candidate_rows)
    )
    prompt = (
        f"Question: {question}\n\n"
        f"Passages:\n{passages}\n\n"
        "Order the passages from the one that best answers the question to the one that answers it least. "
        "Return only a JSON list of the letters, using every letter once. "
        'Example: ["C", "A", "B", "D", "E"]'
    )
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0),  # I use 0 so the answer is repeatable.
    )
    try:
        order = json.loads(re.search(r"\[.*\]", response.text, re.DOTALL).group())
        if sorted(order) == sorted(LETTERS[: len(candidate_rows)]):
            return [candidate_rows[LETTERS.index(letter)] for letter in order], True
    except (AttributeError, json.JSONDecodeError, TypeError, ValueError):
        pass
    # I keep the vector order if the model's reply can't be read.
    return candidate_rows, False


def is_hit(rows, expected_ids, k):
    # I count a hit when at least one expected chunk is in the top k.
    return bool({chunks[row]["chunk_id"] for row in rows[:k]} & expected_ids)


vector_hits = {k: 0 for k in KS}
rerank_hits = {k: 0 for k in KS}
fallbacks = 0
results = []
for position, q in enumerate(questions):
    expected_ids = set(q["expected_chunk_ids"].split(";"))
    candidates = [int(row) for row in vector_rows[position][:CANDIDATES]]
    reranked, ok = rerank(q["question"], candidates)
    if not ok:
        fallbacks += 1

    row_out = {"question_id": q["question_id"], "question_type": q["question_type"],
               "rerank_read_ok": "yes" if ok else "no"}
    for k in KS:
        vector_pass = is_hit(candidates, expected_ids, k)
        rerank_pass = is_hit(reranked, expected_ids, k)
        vector_hits[k] += vector_pass
        rerank_hits[k] += rerank_pass
        row_out[f"Vector_top{k}"] = "yes" if vector_pass else "no"
        row_out[f"Rerank_top{k}"] = "yes" if rerank_pass else "no"
    results.append(row_out)

with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

print(f"Hits out of {len(questions)} questions:")
for label, hits in [("Vector", vector_hits), ("Rerank", rerank_hits)]:
    scores = "   ".join(f"top-{k}: {hits[k]}/{len(questions)}" for k in KS)
    print(f"{label:<8} {scores}")
print(f"Model replies that could not be read (vector order kept): {fallbacks}")
print(f"Per-question results saved to {RESULTS_FILE}")