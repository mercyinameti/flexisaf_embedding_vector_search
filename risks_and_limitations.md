# Risks and Limitations

## The data

- The corpus is 10 sample documents that I wrote myself, with 42 chunks in total. The results show how the methods behave on this small, clean set. They don't show how the methods would perform on real manuals, reports, or policies.
- Because I wrote the documents, the vocabulary is consistent and easy to match. Real documents are messier, use more varied wording, and contain more near-duplicates.
- Chunks are one paragraph each, which suits these documents. Longer or unevenly written documents would need different chunking.

## The evaluation

- The 15 questions and their expected chunk IDs were written by one person, who also wrote the documents. The answer key is a judgement, not a human-verified ground truth.
- For questions with more than one correct chunk, a result counts as a pass if any expected chunk is in the top k. A stricter test would require all of them.
- The re-ranking change was chosen after seeing the failure on Q04, and the same 15 questions were used to judge it. Its effect on new questions has not been measured, so the reported improvement may be optimistic.
- The 15 questions are a small sample. One question changes a score by about seven percentage points.

## The re-ranker

- It uses a language model, so its judgements can vary between runs. Temperature 0 reduces this but does not remove it.
- It only reorders the top 5 vector results. A correct chunk ranked 6th or lower can never be promoted.
- Two of 15 replies could not be parsed in the first run. The script then kept the vector order for those questions, so their results don't reflect re-ranking.
- Each question makes an extra model call. This adds cost and delay, and the free tier has daily request limits.

## The retrieval method

- The keyword search matches exact words only. It does not treat "machine" and "machines," or "learn" and "learns," as the same word. A stemming step would be needed to fix this.
- The metadata filter searches everything and then removes chunks outside the category. That's acceptable for 42 chunks but would be slow for a large collection.
- The vector index is an exact, flat index. It checks every chunk on each search, which is fine at this size but would need a different index type for much larger collections.

## Privacy and cost

- The documents and questions are sent to Google's Gemini API to create embeddings and, for re-ranking, to generate judgements. Do not use confidential or personal documents with this setup unless you have permission to send them to Google.
- The API key is kept in `.env`, which is excluded from Git. The key must never be committed.
- Pricing and free-tier limits can change. The cost assumptions in this project were checked on 6 November 2026.

## Scope

- This system retrieves passages. It does not write answers. Generating an answer from the retrieved passages is the job of a retrieval-augmented generation (RAG) system, which is the next project.