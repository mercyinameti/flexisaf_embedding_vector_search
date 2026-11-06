# FlexiSaf Embedding and Vector Search

A document search system built with Gemini embeddings and FAISS. It splits sample documents into paragraph chunks, stores a vector for each chunk, and returns the chunks closest in meaning to a question. It compares vector, keyword, and hybrid search, filters results by category, and re-ranks the top results with a language model. It was evaluated on 15 questions.

## Setup

Requires Python 3.12.

```
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install rank-bm25
```

Create a file named `.env` in the project folder with your Gemini API key:

```
GEMINI_API_KEY=your-key-here
```

Do not commit `.env`. It is listed in `.gitignore`. `.env.example` shows the required setting without a real key.

## Indexing

Run these in order. The first two create the files the search scripts depend on.

```
python chunk_documents.py
python build_index.py
```

- `chunk_documents.py` reads `sources.csv`, splits each document in `documents/` into paragraphs, and writes `chunks.json`. The first block in each document, its title, is not indexed.
- `build_index.py` sends every chunk to the Gemini embedding model and saves the vectors to `index.faiss`. Run it again whenever the documents change.

`index.faiss` is included in the repo, so you can search without rebuilding. Rebuilding requires a Gemini API key and internet access.

## Usage

Search the whole corpus:

```
python search.py "How do machines learn from examples?"
```

Search one category only:

```
python filter_search.py "Myths" "What is AI?"
```

Compare vector, keyword, and hybrid search:

```
python compare_search.py "What is overfitting?"
```

Compare top-1, top-3, and top-5 results for one question:

```
python top_k_compare.py "How do machines learn from examples?"
```

## Evaluation

`evaluation_questions.csv` holds 15 questions, each with its question type and the chunk IDs that count as correct.

```
python evaluate.py
python rerank_evaluate.py
python diagnose.py Q04
```

- `evaluate.py` scores vector, keyword, and hybrid search and saves per-question results to `evaluation_results.csv`.
- `rerank_evaluate.py` re-ranks the top 5 vector results with a Gemini model and saves results to `rerank_results.csv`.
- `diagnose.py` shows the top results for one question, and where each correct chunk ranks.

A question passes at top-k if at least one of its correct chunks is in the top k results.

## Results

| Method | Top-1 | Top-3 | Top-5 |
|---|---|---|---|
| Vector | 14/15 | 15/15 | 15/15 |
| Keyword | 10/15 | 14/15 | 14/15 |
| Hybrid | 13/15 | 15/15 | 15/15 |
| Vector + re-ranking | 15/15 | 15/15 | 15/15 |

Re-ranking fixed Q04 at top-1. Two model replies could not be read (Q06 and Q14), so vector order was kept for those questions. See `BAD_RESULT.md` for the full record of that failure and fix.

## Files

- `documents/`: 10 sample documents, written for this project
- `sources.csv`: source register with an ID, title, category, and date for each document
- `chunk_documents.py`: splits documents into chunks
- `chunks.json`: the chunks, with their metadata
- `build_index.py`: creates the vector index
- `index.faiss`: the saved vector index
- `search.py`: vector search over all chunks
- `filter_search.py`: vector search within one category
- `compare_search.py`: vector, keyword, and hybrid search side by side
- `top_k_compare.py`: top-1, top-3, and top-5 results for one question
- `evaluation_questions.csv`: 15 evaluation questions and their correct chunk IDs
- `evaluate.py`: scores the three search methods
- `rerank_evaluate.py`: scores re-ranking
- `diagnose.py`: shows the ranking for one question
- `test_embedding.py`: first check that the embedding call works
- `requirements.txt`: pinned package versions
- `.env.example`: template for the API key setting
- `BAD_RESULT.md`: a failed question, the change made, and the outcome
- `risks_and_limitations.md`: known risks and limits

## Limits

See `risks_and_limitations.md`. In short: the corpus is small and written by the project author, the answer key is a judgement, the re-ranking change was chosen after seeing the failure it fixes, and the system returns passages rather than answers.