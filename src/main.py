"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from .recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv")

    # Full taste profile — all 7 features scored
    user_prefs = {
        "favorite_genre":      "lofi",
        "favorite_mood":       "chill",
        "target_energy":       0.38,
        "target_valence":      0.58,
        "target_acousticness": 0.80,
        "target_danceability": 0.58,
        "target_tempo_bpm":    76,
    }

    recommendations = recommend_songs(user_prefs, songs, k=5)

    # ── Header ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  MUSIC RECOMMENDER — Top 5 Results")
    print(f"  Profile: {user_prefs['favorite_genre'].upper()} / "
          f"{user_prefs['favorite_mood'].upper()} / "
          f"energy {user_prefs['target_energy']}")
    print("=" * 60)

    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        # ── Song header ─────────────────────────────────────────────────
        print(f"\n  #{rank}  {song['title']}  —  {song['artist']}")
        print(f"       Score: {score:.2f} / 5.00")
        print(f"       Genre: {song['genre']}  |  Mood: {song['mood']}")

        # ── Reasons — one bullet per feature ────────────────────────────
        print("       Why recommended:")
        for reason in explanation.split(" | "):
            print(f"         • {reason}")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
