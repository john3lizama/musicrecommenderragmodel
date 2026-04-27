"""
Command line runner for the Music Recommender Simulation.

Integrates the RAG (Retrieval-Augmented Generation) layer on top of the
base scoring algorithm. For each recommended song, the RAG layer:
    1. Builds a semantic query from the user profile + song attributes
    2. Retrieves the 2 most relevant music knowledge chunks
    3. Passes them to a local LLM (Ollama llama3.2:3b) to generate
       a warm, natural-language explanation
    4. Falls back to the score-based explanation if Ollama is unavailable

Setup:
    Install Ollama: https://ollama.com
    Pull the model: ollama pull llama3.2:3b
    Start the server: ollama serve
    Then run this file: python -m src.main
"""

import logging

from .recommender import load_songs, recommend_songs
from .rag import KnowledgeBase, OllamaClient, RAGExplainer

# Set to logging.DEBUG to see retrieval scores and full prompts
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s [%(name)s] %(message)s",
)


def main() -> None:
    # ── Load song catalog ────────────────────────────────────────────────────
    songs = load_songs("data/songs.csv")

    # ── Initialize RAG layer (once, shared across all profiles) ─────────────
    # KnowledgeBase loads and embeds the knowledge corpus on first run.
    # Embeddings are cached to data/kb_embeddings.pkl for fast subsequent runs.
    kb      = KnowledgeBase(kb_path="data/knowledge_base.txt")
    ollama  = OllamaClient(model="llama3.2:3b")
    explainer = RAGExplainer(kb=kb, ollama=ollama)

    # ── User profiles to evaluate ────────────────────────────────────────────
    profiles = [
        (
            "High-Energy Pop",
            {
                "favorite_genre":      "pop",
                "favorite_mood":       "happy",
                "target_energy":       0.92,
                "target_valence":      0.88,
                "target_acousticness": 0.12,
                "target_danceability": 0.85,
                "target_tempo_bpm":    140,
            },
        ),
        (
            "Chill Lofi",
            {
                "favorite_genre":      "lofi",
                "favorite_mood":       "chill",
                "target_energy":       0.28,
                "target_valence":      0.58,
                "target_acousticness": 0.92,
                "target_danceability": 0.45,
                "target_tempo_bpm":    70,
            },
        ),
        (
            "Deep Intense Rock",
            {
                "favorite_genre":      "rock",
                "favorite_mood":       "intense",
                "target_energy":       0.95,
                "target_valence":      0.28,
                "target_acousticness": 0.10,
                "target_danceability": 0.60,
                "target_tempo_bpm":    150,
            },
        ),
        (
            "Adversarial — Sad High Energy",
            {
                "favorite_genre":      "rock",
                "favorite_mood":       "sad",
                "target_energy":       0.90,
                "target_valence":      0.10,
                "target_acousticness": 0.20,
                "target_danceability": 0.75,
                "target_tempo_bpm":    170,
            },
        ),
    ]

    # ── Run recommendations for each profile ─────────────────────────────────
    for profile_name, user_prefs in profiles:
        recommendations = recommend_songs(user_prefs, songs, k=5)

        print("\n" + "=" * 60)
        print(f"  MUSIC RECOMMENDER — Top 5 Results")
        print(f"  Profile: {profile_name}")
        print(
            f"  Genre: {user_prefs['favorite_genre'].upper()}"
            f"  |  Mood: {user_prefs['favorite_mood'].upper()}"
            f"  |  Energy: {user_prefs['target_energy']}"
        )
        print("=" * 60)

        for rank, (song, score, score_explanation) in enumerate(recommendations, start=1):
            print(f"\n  #{rank}  {song['title']}  —  {song['artist']}")
            print(f"       Score: {score:.2f} / 5.00")
            print(f"       Genre: {song['genre']}  |  Mood: {song['mood']}")

            # ── RAG explanation ──────────────────────────────────────────────
            # The explainer retrieves relevant knowledge chunks, injects them
            # into a prompt with user context and song info, and calls the local
            # LLM to generate a natural-language explanation. If Ollama is
            # unavailable, the fallback (score_explanation) is used instead.
            rag_explanation = explainer.explain(
                song=song,
                user_prefs=user_prefs,
                fallback=score_explanation,
            )

            # Check whether RAG or fallback was used based on format
            # (RAG output is prose; fallback uses ' | ' separators)
            if " | " in rag_explanation:
                # Fallback: render as bullet points
                print("\n       Why recommended:")
                for reason in rag_explanation.split(" | "):
                    print(f"         • {reason}")
            else:
                # RAG: render as a natural-language paragraph
                print(f"\n       \"{rag_explanation}\"")
                print("\n       Score breakdown:")
                for reason in score_explanation.split(" | "):
                    print(f"         • {reason}")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
