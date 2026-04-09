from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Song:
    """Represents a song and its audio attributes loaded from the CSV catalog."""
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """Stores a listener's target values for genre, mood, energy, and acoustic preference."""
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """OOP wrapper around the catalog that scores and ranks songs for a given user."""

    def __init__(self, songs: List[Song]):
        """Store the song catalog for repeated recommendation calls."""
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top-k songs best matching the user's taste profile."""
        # TODO: Implement recommendation logic
        return self.songs[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Return a human-readable string explaining why this song was recommended."""
        # TODO: Implement explanation logic
        return "Explanation placeholder"

def load_songs(csv_path: str) -> List[Dict]:
    """Parse a CSV file of songs into a list of typed dictionaries."""
    import csv

    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append({
                "id":           int(row["id"]),
                "title":        row["title"],
                "artist":       row["artist"],
                "genre":        row["genre"],
                "mood":         row["mood"],
                "energy":       float(row["energy"]),
                "tempo_bpm":    float(row["tempo_bpm"]),
                "valence":      float(row["valence"]),
                "danceability": float(row["danceability"]),
                "acousticness": float(row["acousticness"]),
            })

    print(f"Loaded songs: {len(songs)}")
    return songs

def _score_song(user_prefs: Dict, song: Dict) -> Tuple[float, str]:
    """Score one song out of 5.0 pts using genre/mood matches and numerical proximity, returning (score, explanation)."""
    score = 0.0
    reasons = []

    # --- Categorical: genre ---
    if song["genre"] == user_prefs.get("favorite_genre", ""):
        score += 1.0
        reasons.append("genre match (+1.0)")
    else:
        reasons.append(f"genre mismatch: {song['genre']} (+0.0)")

    # --- Categorical: mood ---
    if song["mood"] == user_prefs.get("favorite_mood", ""):
        score += 1.0
        reasons.append("mood match (+1.0)")
    else:
        reasons.append(f"mood mismatch: {song['mood']} (+0.0)")

    # --- Numerical: energy (weight 2.00) ---
    if "target_energy" in user_prefs:
        proximity = 1.0 - abs(user_prefs["target_energy"] - song["energy"])
        points = round(2.00 * proximity, 2)
        score += points
        reasons.append(f"energy {song['energy']} vs target {user_prefs['target_energy']} (+{points})")

    # --- Numerical: valence (weight 0.50) ---
    if "target_valence" in user_prefs:
        proximity = 1.0 - abs(user_prefs["target_valence"] - song["valence"])
        points = round(0.50 * proximity, 2)
        score += points
        reasons.append(f"valence {song['valence']} vs target {user_prefs['target_valence']} (+{points})")

    # --- Numerical: acousticness (weight 0.30) ---
    if "target_acousticness" in user_prefs:
        proximity = 1.0 - abs(user_prefs["target_acousticness"] - song["acousticness"])
        points = round(0.30 * proximity, 2)
        score += points
        reasons.append(f"acousticness {song['acousticness']} vs target {user_prefs['target_acousticness']} (+{points})")

    # --- Numerical: danceability (weight 0.15) ---
    if "target_danceability" in user_prefs:
        proximity = 1.0 - abs(user_prefs["target_danceability"] - song["danceability"])
        points = round(0.15 * proximity, 2)
        score += points
        reasons.append(f"danceability {song['danceability']} vs target {user_prefs['target_danceability']} (+{points})")

    # --- Numerical: tempo_bpm (weight 0.05, normalized to 0–1) ---
    if "target_tempo_bpm" in user_prefs:
        BPM_MIN, BPM_MAX = 60, 200
        norm_song = (song["tempo_bpm"] - BPM_MIN) / (BPM_MAX - BPM_MIN)
        norm_user = (user_prefs["target_tempo_bpm"] - BPM_MIN) / (BPM_MAX - BPM_MIN)
        proximity = 1.0 - abs(norm_user - norm_song)
        points = round(0.05 * proximity, 2)
        score += points
        reasons.append(f"tempo {song['tempo_bpm']} BPM vs target {user_prefs['target_tempo_bpm']} BPM (+{points})")

    explanation = " | ".join(reasons)
    return round(score, 2), explanation


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences.
    Required by recommend_songs() and src/main.py
    """
    # TODO: Implement scoring logic using your Algorithm Recipe from Phase 2.
    # Expected return format: (score, reasons)
    return []

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Score all songs, sort by score descending, and return the top-k as (song, score, explanation) tuples."""
    scored = [(_score_song(user_prefs, song), song) for song in songs]
    scored.sort(key=lambda x: x[0][0], reverse=True)
    return [(song, score, explanation) for (score, explanation), song in scored[:k]]
