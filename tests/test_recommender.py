import os
import tempfile
from unittest.mock import MagicMock, patch

from src.recommender import Song, UserProfile, Recommender
from src.rag import KnowledgeBase, OllamaClient, RAGExplainer


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    # Starter expectation: the pop, happy, high energy song should score higher
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


# ══════════════════════════════════════════════════════════════════════════════
# RAG Tests — all run offline, no Ollama required
# ══════════════════════════════════════════════════════════════════════════════

def make_temp_kb(chunks: list[str]) -> str:
    """Write a list of text chunks to a temp file and return its path."""
    content = "\n---\n".join(chunks)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    )
    tmp.write(content)
    tmp.close()
    return tmp.name


# ── Test 1: Retrieval results are sorted by similarity score descending ──────

def test_retrieval_results_sorted_by_score():
    """
    retrieve() must return results in descending similarity order.
    The chunk most semantically similar to the query should come first.
    """
    chunks = [
        "Lofi hip-hop is mellow, warm, and perfect for late-night studying.",
        "Metal is loud, aggressive, and defined by distorted guitars at high tempo.",
        "Classical music is acoustic, slow, and emotionally complex.",
    ]
    kb_path = make_temp_kb(chunks)
    try:
        kb = KnowledgeBase(kb_path=kb_path, cache_path=kb_path + ".pkl")
        results = kb.retrieve("chill lofi studying", top_k=3)

        assert len(results) == 3, "Should return exactly 3 results"
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True), (
            f"Results should be sorted descending by score, got: {scores}"
        )
    finally:
        os.unlink(kb_path)
        if os.path.exists(kb_path + ".pkl"):
            os.unlink(kb_path + ".pkl")


# ── Test 2: Semantic similarity — "chill lofi" retrieves the lofi chunk ──────

def test_semantic_similarity_lofi_query():
    """
    A 'chill lofi' query should retrieve the lofi chunk as the top result,
    not the metal or classical chunk, demonstrating semantic understanding.
    """
    chunks = [
        "Lofi hip-hop is mellow, warm, and perfect for late-night studying.",
        "Metal is loud, aggressive, and defined by distorted guitars at high tempo.",
        "Classical music is acoustic, slow, and emotionally complex.",
    ]
    kb_path = make_temp_kb(chunks)
    try:
        kb = KnowledgeBase(kb_path=kb_path, cache_path=kb_path + ".pkl")
        results = kb.retrieve("chill lofi", top_k=1)

        assert len(results) == 1
        top_chunk, top_score = results[0]
        assert "lofi" in top_chunk.lower(), (
            f"Expected lofi chunk to be top result, got: '{top_chunk[:60]}'"
        )
    finally:
        os.unlink(kb_path)
        if os.path.exists(kb_path + ".pkl"):
            os.unlink(kb_path + ".pkl")


# ── Test 3: Empty knowledge base does not crash ───────────────────────────────

def test_empty_knowledge_base_does_not_crash():
    """
    If the knowledge base file is empty (or has only comments),
    KnowledgeBase should log an error but not raise an exception.
    retrieve() should return an empty list gracefully.
    """
    kb_path = make_temp_kb(["# This is just a comment"])
    try:
        kb = KnowledgeBase(kb_path=kb_path, cache_path=kb_path + ".pkl")
        results = kb.retrieve("lofi chill", top_k=2)
        assert results == [], f"Expected empty list, got: {results}"
    finally:
        os.unlink(kb_path)
        if os.path.exists(kb_path + ".pkl"):
            os.unlink(kb_path + ".pkl")


# ── Test 4: RAGExplainer falls back when Ollama returns None ─────────────────

def test_explainer_falls_back_when_ollama_returns_none():
    """
    If OllamaClient.generate() returns None (e.g. Ollama not running),
    RAGExplainer.explain() must return the fallback string instead of crashing.
    """
    chunks = ["Lofi hip-hop is mellow and warm, ideal for late nights."]
    kb_path = make_temp_kb(chunks)
    try:
        kb = KnowledgeBase(kb_path=kb_path, cache_path=kb_path + ".pkl")

        # Mock OllamaClient so no real HTTP call is made
        mock_ollama = MagicMock(spec=OllamaClient)
        mock_ollama.generate.return_value = None

        explainer = RAGExplainer(kb=kb, ollama=mock_ollama)

        song       = {"title": "Library Rain", "artist": "Paper Lanterns",
                      "genre": "lofi",         "mood": "chill"}
        user_prefs = {"favorite_genre": "lofi", "favorite_mood": "chill"}
        fallback   = "genre match (+1.0) | mood match (+1.0) | energy 0.35 vs target 0.28 (+1.86)"

        result = explainer.explain(song=song, user_prefs=user_prefs, fallback=fallback)

        assert result == fallback, (
            f"Expected fallback string, got: '{result}'"
        )
        mock_ollama.generate.assert_called_once()
    finally:
        os.unlink(kb_path)
        if os.path.exists(kb_path + ".pkl"):
            os.unlink(kb_path + ".pkl")


# ── Test 5: Quality guard rejects responses that are too short ───────────────

def test_quality_guard_rejects_short_response():
    """
    If Ollama returns a response shorter than 10 characters,
    the quality guard should reject it and use the fallback instead.
    """
    chunks = ["Lofi hip-hop is mellow and warm, ideal for late nights."]
    kb_path = make_temp_kb(chunks)
    try:
        kb = KnowledgeBase(kb_path=kb_path, cache_path=kb_path + ".pkl")

        mock_ollama = MagicMock(spec=OllamaClient)
        mock_ollama.generate.return_value = "ok"   # 2 chars — below the 10-char minimum

        explainer = RAGExplainer(kb=kb, ollama=mock_ollama)

        song       = {"title": "Library Rain", "artist": "Paper Lanterns",
                      "genre": "lofi",         "mood": "chill"}
        user_prefs = {"favorite_genre": "lofi", "favorite_mood": "chill"}
        fallback   = "genre match (+1.0) | mood match (+1.0)"

        result = explainer.explain(song=song, user_prefs=user_prefs, fallback=fallback)

        assert result == fallback, (
            f"Expected fallback due to short response, got: '{result}'"
        )
    finally:
        os.unlink(kb_path)
        if os.path.exists(kb_path + ".pkl"):
            os.unlink(kb_path + ".pkl")
