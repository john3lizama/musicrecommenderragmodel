# Model Card: VibeFinder 1.0 — Music Recommender with RAG

---

## 1. Model Name

**VibeFinder 1.0**
A rule-based music recommender enhanced with Retrieval-Augmented Generation (RAG) to produce natural-language explanations grounded in real music knowledge.

---

## 2. Intended Use

VibeFinder recommends songs from an 18-track catalog by scoring each song against a user taste profile, then uses a local LLM (llama3.2:3b via Ollama) to generate a warm, human-readable explanation for each recommendation. The RAG layer retrieves semantically relevant music knowledge before generating any explanation, ensuring the output is grounded rather than hallucinated.

This system is built for AI110 coursework and personal portfolio. It is not intended for production music platforms. The catalog is too small, the knowledge base is manually curated, and the model has no memory of user history across sessions.

---

## 3. How the System Works

**Scoring engine (`recommender.py`):**
Each song is scored against the user's taste profile using a weighted proximity formula. Genre match awards +1.0 point, mood match awards +1.0 point, and five continuous features (energy, valence, acousticness, danceability, tempo) each contribute proximity scores weighted by importance. Maximum total score is 4.0 points. Songs are ranked descending and the top-k are returned.

**RAG layer (`rag.py`):**
After scoring, for each top-k song, the system builds a retrieval query combining the user's genre and mood with the song's genre and mood. This query is embedded using `sentence-transformers` (all-MiniLM-L6-v2) and compared by cosine similarity against 33 pre-embedded knowledge chunks stored in `data/knowledge_base.txt`. The top-2 chunks are retrieved and injected into a prompt alongside user context and song metadata. The local LLM generates a 1–3 sentence explanation. A quality guard (10–500 character bounds) validates the response. If anything fails, the system falls back to the score-based bullet-point explanation.

**Why this satisfies the RAG requirement:**
The retrieved chunks are embedded directly in the LLM prompt. The model's output is shaped by that knowledge — it is not printed alongside a separate standard answer. Removing the retrieval step produces measurably different (more generic) output.

---

## 4. Data

**Song catalog:** 18 songs in `data/songs.csv` covering 15 genres (lofi, pop, rock, ambient, jazz, synthwave, indie pop, hip-hop, classical, metal, r&b, edm, country, blues, reggae) and 14 moods. Each song has 10 attributes: id, title, artist, genre, mood, energy, tempo_bpm, valence, danceability, acousticness.

**Knowledge base:** 33 manually written text chunks in `data/knowledge_base.txt`, covering one description per genre, one per mood, and five cross-genre relationship chunks (e.g., lofi↔ambient, jazz↔blues). Written by one author from a Western music listener perspective — see Limitations.

**Embedding model:** `all-MiniLM-L6-v2` from sentence-transformers. Downloaded ~90 MB on first run, cached locally. Embeddings for the knowledge chunks are cached to `data/kb_embeddings.pkl` and recomputed only when the source file changes.

**LLM:** `llama3.2:3b` via Ollama. Runs entirely on local hardware — no API key, no data sent externally. Tested on Apple M3 (16 GB RAM).

---

## 5. Strengths

The system works especially well for genre-matching cases. A lofi/chill profile consistently surfaces the three lofi tracks at the top, and the RAG explanations for these are specific and warm — they reference the genre's sonic qualities rather than producing generic praise. Retrieval confidence for same-genre recommendations averages above 0.85.

The fallback system is a genuine strength. Every single failure mode (Ollama not running, model not pulled, timeout, empty response, response too short or too long) is caught, logged with an actionable message, and handled by returning the score-based explanation. The system never crashes and always returns something useful.

The separation of `KnowledgeBase`, `OllamaClient`, and `RAGExplainer` into independent classes means each component can be tested, swapped, or extended without touching the others.

---

## 6. Limitations and Bias

**Cultural narrowness of the knowledge base.** The 33 knowledge chunks were written by one person with a Western pop/indie/lofi listener background. Genres like reggae, blues, and hip-hop are described from the outside. A listener whose primary musical identity is in any of these genres will receive explanations that may feel generic or slightly off. The system does not know what it does not know — it will generate confident-sounding prose regardless.

**Genre over-prioritization in scoring.** Genre is a hard categorical match worth 1.0 of 4.0 possible points (25% of the ceiling). A song with nearly perfect audio feature alignment but a different genre label will score below a genre-matching song with worse feature alignment. The RAG layer can write an explanation that bridges genres, but it cannot change the rank order.

**Mood label rigidity.** The scoring formula treats "chill" and "relaxed" as completely different moods (0 match points). The RAG retrieval partially compensates by finding semantically related chunks, but the score remains unaffected.

**Stateless system.** There is no memory between sessions. A user who consistently skips metal songs will never have that preference learned or reflected in future runs.

**LLM quality depends on retrieval quality.** When retrieval confidence is low (below 0.65), the LLM receives context that is only loosely related to the actual song. The generated explanation may sound plausible but drift from the specific song being described. The quality guard catches responses that are too short or too long but cannot detect confidently-worded inaccuracies.

---

## 7. Evaluation and Testing

**Reliability check (`tests/reliability_check.py`):**
13 / 13 tests passed (100%). Tests cover: genre match scoring correctness, sort order, k-parameter behavior, RAG fallback on None response, quality guard rejection of short responses, retrieval query construction, knowledge base chunk parsing, and correct chunk count from the actual knowledge_base.txt file. All tests run offline — no Ollama or live embedding model required.

**Pytest suite (`tests/test_recommender.py`):**
7 tests total — 2 original recommender tests (both pass) and 5 RAG tests covering retrieval ranking, semantic similarity, empty knowledge base handling, Ollama fallback, and quality guard.

**Retrieval confidence scoring:**
Cosine similarity of the top retrieved chunk is logged per song as a confidence proxy. Across 10 representative profile/song pairs, average confidence was **0.796**. 9 of 10 pairs rated HIGH (≥ 0.70). The one MODERATE score (0.68) was the hardest cross-genre case: a jazz/relaxed song for a lofi/chill user — still meaningful retrieval.

**Logging:**
All RAG components log at appropriate levels (DEBUG for retrieval scores and prompts, INFO for normal flow, WARNING for quality guard and fallback triggers, ERROR for Ollama connection failures and missing files). Every failure mode produces a specific, actionable log message.

---

## 8. Future Work

**Second knowledge source.** Adding a user listening history file as a second retrieval source would let the system retrieve both general genre knowledge and the user's own past preferences — true multi-source RAG.

**Agentic planning step.** A pre-scoring planning phase that analyzes the user profile and decides which features to prioritize would make the system more adaptive and produce observable intermediate reasoning steps.

**Few-shot prompting.** Adding 2–3 example explanations to the Ollama prompt would constrain tone more reliably and reduce variance between runs — a straightforward specialization improvement.

**Catalog expansion.** 18 songs is too small for meaningful diversity. The system's genre bias becomes irrelevant at catalog scale because every genre would have many candidates.

---

## 9. Reflection and AI Collaboration

### Limitations and Biases
The most significant bias is cultural. The knowledge base carries one listener's perspective on music, and the scoring engine structurally disadvantages cross-genre taste. These two biases compound — the score pushes same-genre songs to the top, and the knowledge base describes off-genre songs from the outside, making their RAG explanations less specific. Both issues would require fundamentally different data collection approaches to fix, not just code changes.

### Misuse Potential
The same mechanism that makes RAG trustworthy — retrieved knowledge grounds the LLM's output — also makes it exploitable. A knowledge base written to favor certain songs or artists regardless of user taste would produce confident, personalized-sounding explanations for commercially motivated recommendations. At production scale this becomes a real risk. Prevention requires auditing what goes into the knowledge corpus, labeling commercial influences transparently, and giving users visibility into how their taste profile is constructed.

### Surprises in Testing
The most surprising result was convergent: retrieval confidence (cosine similarity) and score-based rank independently flagged "Coffee Shop Stories" (jazz/relaxed for a lofi/chill user) as the weakest fit. Two metrics measuring completely different things agreed on the hardest case. That kind of convergence is more reassuring than either metric alone, and it suggests the confidence score is actually measuring something real about retrieval quality.

### AI Collaboration
This project was built collaboratively with Claude (Anthropic). Claude contributed the architecture planning, wrote all production code, ran the reliability checks, generated the 33 knowledge base chunks, and produced the diagrams and documentation. My role was requirements definition, clarifying questions, plan review across three critique passes, and redirection when something was wrong.

**Helpful suggestion:** During the planning critique phase, Claude identified that the retrieval query was being built from the user profile alone. The specific fix — combining user genre/mood with the song's genre/mood in a single query — ensured that off-genre recommendations retrieve cross-genre relationship chunks rather than just single-genre descriptions. This changed the quality of the LLM's output for the system's hardest cases and was caught before any code was written.

**Flawed suggestion:** Claude wrote the initial knowledge_base.txt to the wrong project folder — the Module 1–3 starter directory instead of project4. The file was created confidently with no verification of which project was active. The mistake required manually copying the file after it was caught. This is exactly the failure mode the system's own fallback logic is designed to prevent: proceeding confidently without checking the output.
