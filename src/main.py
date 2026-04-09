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

    for profile_name, user_prefs in profiles:
        recommendations = recommend_songs(user_prefs, songs, k=5)

        print("\n" + "=" * 60)
        print(f"  MUSIC RECOMMENDER — Top 5 Results")
        print(f"  Profile: {profile_name}")
        print(f"  Genre: {user_prefs['favorite_genre'].upper()}  |  Mood: {user_prefs['favorite_mood'].upper()}  |  Energy: {user_prefs['target_energy']}")
        print("=" * 60)

        for rank, (song, score, explanation) in enumerate(recommendations, start=1):
            print(f"\n  #{rank}  {song['title']}  —  {song['artist']}")
            print(f"       Score: {score:.2f} / 5.00")
            print(f"       Genre: {song['genre']}  |  Mood: {song['mood']}")
            print("       Why recommended:")
            for reason in explanation.split(" | "):
                print(f"         • {reason}")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
