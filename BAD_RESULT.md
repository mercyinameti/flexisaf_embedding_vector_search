# Bad Result: Question Q04

## The question

"What is the difference between narrow AI and general AI?"

Expected chunks: `04-001` (narrow AI) and `04-002` (general AI). Either one counts as a correct answer.

## What went wrong

Vector search missed this question at top-1. Its first result was `04-003`, which says "The difference matters because the two are often confused in news and conversation." That chunk shares the words "difference," "narrow," and "general" with the question, but it doesn't define either kind of AI.

The two correct chunks were ranked second (`04-001`) and fourth (`04-002`). Keyword search and hybrid search also ranked `04-003` first.

The question passed at top-3 for vector search, because `04-001` was in the top 3.

## The change

I added a re-ranking step. Vector search returns its top 5 chunks, and the Gemini model `gemini-3.5-flash-lite` (temperature 0) orders those 5 by how well each one answers the question. Only the order of those 5 changes. The model can't bring in a chunk from outside the top 5.

## The rerun

I reran all 15 evaluation questions.

- **Q04:** top-1 went from a miss to a hit.
- **Top-3 and top-5:** 15/15 for both vector search and re-ranking.
- **Top-1 overall:** vector 14/15, re-ranked 15/15.
- **Two replies could not be read** (Q06 and Q14). For those, the script kept the vector order, so they count as passes without re-ranking.

## Outcome

The re-ranking fixed Q04 at top-1 and did not break any question that passed before. On the 13 questions where the model's reply could be read, re-ranking got all 13 right at top-1.

## Limits of this result

- The change was chosen after seeing Q04, and the same 15 questions were used to judge it. Its effect on new questions is not yet measured.
- Two model replies could not be read, so those questions were not re-ranked.
- The model's judgement was compared with my expected chunk IDs, not with human judgements.
- Each question makes an extra model call, which adds cost and delay.