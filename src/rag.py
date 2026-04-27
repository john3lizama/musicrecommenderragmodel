"""
src/rag.py — Retrieval-Augmented Generation (RAG) layer for the Music Recommender.

Architecture:
    KnowledgeBase  — loads, validates, embeds, caches, and retrieves music knowledge chunks
    OllamaClient   — wraps the Ollama HTTP API with explicit, actionable error handling
    RAGExplainer   — orchestrates retrieve → prompt → generate → quality check → fallback

Flow per song recommendation:
    user_prefs + song
        → RAGExplainer._build_query()          # combine user taste + song attributes
        → KnowledgeBase.retrieve(query, k=2)   # cosine-similarity search over knowledge chunks
        → RAGExplainer._build_prompt()         # inject user context + chunks + song info
        → OllamaClient.generate(prompt)        # local LLM call via Ollama
        → RAGExplainer._quality_check()        # guard against empty or runaway responses
        → return explanation (or fallback)     # always returns a string, never raises

Why RAG here?
    The base recommender uses rigid string matching — "chill" and "relaxed" are treated as
    completely different. RAG solves this by retrieving semantically relevant music knowledge
    before generating an explanation, so the LLM can write grounded, natural descriptions
    that bridge genres and moods the scoring formula treats as unrelated.
"""

import json
import logging
import os
import pickle
from typing import Optional

import numpy as np
import requests

# ── Logging setup ────────────────────────────────────────────────────────────
# Each class uses its own named logger so log output is easy to filter.
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s [%(name)s] %(message)s",
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. KnowledgeBase
# ══════════════════════════════════════════════════════════════════════════════

class KnowledgeBase:
    """
    Manages the music knowledge corpus used for RAG retrieval.

    Responsibilities:
        - Parse a plain-text file of music descriptions into chunks
        - Validate the corpus has enough content to be useful
        - Embed all chunks using sentence-transformers (all-MiniLM-L6-v2)
        - Cache embeddings to disk and reload them if the source file is unchanged
        - Expose a retrieve() method for cosine-similarity search

    Knowledge file format:
        Chunks are separated by lines containing only '---'.
        Lines starting with '#' are treated as comments and skipped.

    Example chunk:
        Lofi hip-hop wraps you in quiet comfort. Its mellow tempos and warm
        textures are ideal for studying or winding down late at night.
    """

    _EMBED_MODEL = "all-MiniLM-L6-v2"   # ~90 MB, downloaded once on first run
    _MIN_CHUNKS  = 3                      # warn if corpus is smaller than this

    def __init__(self, kb_path: str, cache_path: str = "data/kb_embeddings.pkl"):
        self.log        = logging.getLogger("rag.kb")
        self.kb_path    = kb_path
        self.cache_path = cache_path
        self.chunks: list[str]            = []
        self.embeddings: Optional[np.ndarray] = None
        self._model     = None            # lazy-loaded on first encode call

        self._load_chunks()
        self._validate()
        self._load_or_embed()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_chunks(self) -> None:
        """Parse the knowledge base file into a list of clean text chunks."""
        if not os.path.exists(self.kb_path):
            self.log.error(
                f"Knowledge base file not found: '{self.kb_path}'. "
                "RAG will not function until this file exists."
            )
            return

        with open(self.kb_path, "r", encoding="utf-8") as f:
            raw = f.read()

        chunks = []
        for block in raw.split("---"):
            # Drop comment lines and blank lines, join the rest into one string
            lines = [
                line for line in block.splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            text = " ".join(lines).strip()
            if text:
                chunks.append(text)

        self.chunks = chunks
        self.log.info(f"Loaded {len(self.chunks)} knowledge chunks from '{self.kb_path}'")

    def _validate(self) -> None:
        """Warn if the knowledge corpus is too small to provide useful retrieval."""
        if len(self.chunks) == 0:
            self.log.error(
                "Knowledge base is empty — RAG retrieval will return no results. "
                "Add content to the knowledge base file."
            )
        elif len(self.chunks) < self._MIN_CHUNKS:
            self.log.warning(
                f"Knowledge base has only {len(self.chunks)} chunk(s) "
                f"(minimum recommended: {self._MIN_CHUNKS}). "
                "Add more chunks for better retrieval quality."
            )

    def _get_model(self):
        """Lazy-load the sentence-transformer embedding model."""
        if self._model is None:
            self.log.info(
                f"Loading embedding model '{self._EMBED_MODEL}'. "
                "First run will download ~90 MB — subsequent runs use a local cache."
            )
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._EMBED_MODEL)
            self.log.info("Embedding model ready.")
        return self._model

    def _load_or_embed(self) -> None:
        """
        Load chunk embeddings from the disk cache if the knowledge file is unchanged,
        otherwise recompute embeddings and save them to cache.

        Cache invalidation: if the knowledge file's modification time has changed
        since the cache was written, the cache is considered stale and recomputed.
        """
        if not self.chunks:
            return

        kb_mtime = os.path.getmtime(self.kb_path)

        # Attempt to load from cache
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    cache = pickle.load(f)
                cached_mtime      = cache.get("mtime")
                cached_embeddings = cache.get("embeddings")
                if (
                    cached_mtime == kb_mtime
                    and cached_embeddings is not None
                    and len(cached_embeddings) == len(self.chunks)
                ):
                    self.embeddings = cached_embeddings
                    self.log.info(f"Embeddings loaded from cache ('{self.cache_path}')")
                    return
                else:
                    self.log.info("Cache is stale — recomputing embeddings.")
            except Exception as exc:
                self.log.warning(f"Could not read embedding cache ({exc}) — recomputing.")

        # Recompute embeddings
        self.log.info(f"Computing embeddings for {len(self.chunks)} chunks...")
        model = self._get_model()
        self.embeddings = model.encode(self.chunks, convert_to_numpy=True)
        self.log.info("Embeddings computed.")

        # Save to cache
        try:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump({"mtime": kb_mtime, "embeddings": self.embeddings}, f)
            self.log.info(f"Embeddings cached to '{self.cache_path}'")
        except Exception as exc:
            self.log.warning(f"Could not write embedding cache: {exc}")

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 2) -> list[tuple[str, float]]:
        """
        Embed the query string and return the top_k most semantically similar
        knowledge chunks, sorted by cosine similarity descending.

        Args:
            query:  Natural language query (e.g. "lofi chill ambient peaceful")
            top_k:  Number of chunks to return

        Returns:
            List of (chunk_text, similarity_score) tuples, highest score first.
            Returns an empty list if the knowledge base is unavailable.
        """
        if not self.chunks or self.embeddings is None:
            self.log.warning("Knowledge base unavailable — returning no retrieval results.")
            return []

        model     = self._get_model()
        query_vec = model.encode([query], convert_to_numpy=True)[0]

        # Cosine similarity: dot(chunk, query) / (|chunk| * |query|)
        chunk_norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm  = np.linalg.norm(query_vec)
        denominators = chunk_norms * query_norm
        denominators = np.where(denominators == 0, 1e-9, denominators)   # avoid div/0
        scores = (self.embeddings @ query_vec) / denominators

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(self.chunks[i], float(scores[i])) for i in top_indices]

        self.log.debug(
            f"Retrieved {len(results)} chunk(s) for query '{query[:60]}' "
            + " | ".join(f"score={s:.3f}" for _, s in results)
        )
        return results


# ══════════════════════════════════════════════════════════════════════════════
# 2. OllamaClient
# ══════════════════════════════════════════════════════════════════════════════

class OllamaClient:
    """
    Thin, fault-tolerant wrapper around the Ollama local LLM HTTP API.

    Design principles:
        - Never raises an exception — always returns a string or None
        - Every failure mode is caught and logged with a clear, actionable message
        - The caller (RAGExplainer) decides what to do with a None response

    Requires Ollama to be running locally:
        ollama serve                  # start the Ollama server
        ollama pull llama3.2:3b       # pull the model (one-time, ~2 GB)
    """

    _API_URL = "http://localhost:11434/api/generate"
    _TIMEOUT = 30   # seconds before giving up on a slow response

    def __init__(self, model: str = "llama3.2:3b"):
        self.model = model
        self.log   = logging.getLogger("rag.ollama")

    def generate(self, prompt: str) -> Optional[str]:
        """
        Send a prompt to Ollama and return the generated text.

        Args:
            prompt: The full prompt string to send to the model.

        Returns:
            The model's response as a string, or None if the call fails for any reason.

        Handled error cases:
            ConnectionError  → Ollama is not running
            HTTP 404         → The requested model has not been pulled yet
            Timeout          → The model took too long to respond
            JSONDecodeError  → Ollama returned a malformed response
            Other HTTPError  → Any other HTTP-level failure
        """
        payload = {
            "model":  self.model,
            "prompt": prompt,
            "stream": False,
        }

        try:
            self.log.info(f"Calling Ollama ({self.model})...")
            response = requests.post(self._API_URL, json=payload, timeout=self._TIMEOUT)

            # Model not pulled yet — give the user the exact command to fix it
            if response.status_code == 404:
                self.log.error(
                    f"Model '{self.model}' not found. "
                    f"Pull it with: ollama pull {self.model}"
                )
                return None

            response.raise_for_status()

            data = response.json()
            text = data.get("response", "").strip()

            if not text:
                self.log.warning("Ollama returned an empty response.")
                return None

            self.log.info("Ollama response received successfully.")
            return text

        except requests.exceptions.ConnectionError:
            self.log.error(
                "Cannot connect to Ollama. "
                "Start it with: ollama serve"
            )
            return None

        except requests.exceptions.Timeout:
            self.log.warning(
                f"Ollama did not respond within {self._TIMEOUT}s. "
                "The model may be loading — try again in a moment."
            )
            return None

        except json.JSONDecodeError:
            self.log.warning("Ollama returned a response that could not be parsed as JSON.")
            return None

        except requests.exceptions.HTTPError as exc:
            self.log.error(f"Ollama HTTP error: {exc}")
            return None


# ══════════════════════════════════════════════════════════════════════════════
# 3. RAGExplainer
# ══════════════════════════════════════════════════════════════════════════════

class RAGExplainer:
    """
    Orchestrates the full RAG pipeline for generating song recommendation explanations.

    For each song in the top-k results, RAGExplainer:
        1. Builds a retrieval query combining user taste + song attributes
        2. Retrieves the 2 most semantically relevant knowledge chunks
        3. Constructs a grounded LLM prompt (user context + chunks + song info)
        4. Calls OllamaClient to generate a natural-language explanation
        5. Runs a quality check (response length bounds)
        6. Falls back to the original score-based explanation if anything goes wrong

    This design satisfies the RAG requirement: retrieved knowledge is injected into
    the prompt and actively shapes the LLM's response — it is not merely printed
    alongside a standard answer.

    Dependencies are injected so each class can be tested independently.
    """

    _MIN_LEN = 10    # characters — responses shorter than this are rejected
    _MAX_LEN = 500   # characters — responses longer than this are rejected

    def __init__(self, kb: KnowledgeBase, ollama: OllamaClient):
        self.kb     = kb
        self.ollama = ollama
        self.log    = logging.getLogger("rag.explainer")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_query(self, song: dict, user_prefs: dict) -> str:
        """
        Build a retrieval query that captures both the user's taste and the
        specific song being explained.

        Using both sides of the match (user genre/mood + song genre/mood) ensures
        the retrieved chunks are relevant to the relationship between them — not just
        the user's preference in isolation. This is especially important for off-genre
        recommendations where we need context about why two genres feel similar.

        Example:
            user: lofi / chill  +  song: ambient / chill
            query: "lofi chill ambient chill"
            → retrieves the lofi-ambient crossover chunk
        """
        return (
            f"{user_prefs.get('favorite_genre', '')} "
            f"{user_prefs.get('favorite_mood', '')} "
            f"{song.get('genre', '')} "
            f"{song.get('mood', '')}"
        ).strip()

    def _build_prompt(
        self,
        song: dict,
        user_prefs: dict,
        chunks: list[tuple[str, float]],
    ) -> str:
        """
        Construct the full LLM prompt.

        The retrieved knowledge chunks are embedded directly in the prompt so the
        model's response is grounded in real music knowledge rather than generic
        associations from pre-training. This is what makes it RAG rather than
        just a prompted LLM call.
        """
        chunk_text  = "\n\n".join(f"- {chunk}" for chunk, _ in chunks)
        user_genre  = user_prefs.get("favorite_genre", "unknown")
        user_mood   = user_prefs.get("favorite_mood",  "unknown")
        song_title  = song.get("title",  "Unknown")
        song_artist = song.get("artist", "Unknown")
        song_genre  = song.get("genre",  "unknown")
        song_mood   = song.get("mood",   "unknown")

        return (
            f"You are a music-savvy friend making a personal recommendation. "
            f"Write 1 to 3 sentences explaining why a song fits someone's vibe. "
            f"Sound warm and natural, like a friend who really knows their taste. "
            f"Never mention scores, numbers, or technical audio terms.\n\n"
            f"The listener enjoys {user_genre} music with a {user_mood} mood.\n\n"
            f"Here is some relevant music knowledge to inform your response:\n"
            f"{chunk_text}\n\n"
            f'The song is "{song_title}" by {song_artist}. '
            f"It is {song_genre} with a {song_mood} mood.\n\n"
            f"In 1 to 3 sentences, why does this song fit their vibe?"
        )

    def _quality_check(self, response: str) -> bool:
        """
        Verify the LLM response meets minimum quality standards.

        Rejects responses that are:
            - Too short  (< 10 chars): likely an error or empty reply
            - Too long   (> 500 chars): model went off-script or repeated itself

        Returns True if the response is acceptable, False to trigger fallback.
        """
        length = len(response.strip())
        if length < self._MIN_LEN:
            self.log.warning(
                f"LLM response too short ({length} chars) — falling back to score explanation."
            )
            return False
        if length > self._MAX_LEN:
            self.log.warning(
                f"LLM response too long ({length} chars) — falling back to score explanation."
            )
            return False
        return True

    # ── Public API ────────────────────────────────────────────────────────────

    def explain(self, song: dict, user_prefs: dict, fallback: str) -> str:
        """
        Generate a natural-language explanation for why a song was recommended.

        The explanation is grounded in retrieved music knowledge and generated
        by a local LLM. If any step fails, the original score-based explanation
        is returned instead — the user always gets something meaningful.

        Args:
            song:       Song dict from the catalog (keys: title, artist, genre, mood, etc.)
            user_prefs: User preference dict (keys: favorite_genre, favorite_mood, etc.)
            fallback:   The existing pipe-separated score explanation to use if RAG fails

        Returns:
            A natural-language string (LLM-generated or fallback). Never raises.
        """
        # Step 1: Build retrieval query from both user profile and song attributes
        query = self._build_query(song, user_prefs)
        self.log.debug(f"Retrieval query for '{song.get('title')}': '{query}'")

        # Step 2: Retrieve relevant knowledge chunks
        chunks = self.kb.retrieve(query, top_k=2)
        if not chunks:
            self.log.warning(
                f"No chunks retrieved for '{song.get('title')}' — using fallback."
            )
            return fallback

        # Step 3: Build the grounded prompt
        prompt = self._build_prompt(song, user_prefs, chunks)

        # Step 4: Call the local LLM
        response = self.ollama.generate(prompt)
        if response is None:
            self.log.warning(
                f"Ollama returned no response for '{song.get('title')}' — using fallback."
            )
            return fallback

        # Step 5: Quality check
        if not self._quality_check(response):
            return fallback

        self.log.info(f"RAG explanation generated for '{song.get('title')}'")
        return response
