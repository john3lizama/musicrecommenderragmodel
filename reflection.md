# Reflection

---

## Profile Comparison Notes

### High-Energy Pop vs Chill Lofi
The High-Energy Pop profile surfaces upbeat, danceable tracks like "Sunrise City" with high energy and pop vibes, while Chill Lofi shifts to mellow, acoustic lofi songs like "Library Rain" with low energy and chill moods. This makes sense as energy is the key differentiator, with pop favoring high-intensity party music and lofi emphasizing relaxed study beats.

### High-Energy Pop vs Deep Intense Rock
High-Energy Pop recommends happy, mid-tempo pop songs like "Gym Hero," whereas Deep Intense Rock prioritizes aggressive, high-energy rock like "Storm Runner" with intense moods. The difference highlights how mood (happy vs intense) and genre create distinct sonic profiles, even with similar energy levels.

### High-Energy Pop vs Adversarial (Sad High Energy)
Both profiles favor high-energy songs, but High-Energy Pop ranks pop tracks with happy moods first, while the adversarial profile (sad mood) still boosts rock songs due to genre, despite mood penalties. This shows the system's genre bias overpowering mood conflicts, making the adversarial results feel mismatched for "sad" preferences.

### Chill Lofi vs Deep Intense Rock
Chill Lofi recommends low-energy, acoustic lofi and ambient tracks like "Spacewalk Thoughts," contrasting with Deep Intense Rock's high-energy, non-acoustic rock like "Storm Runner." The energy gap (low vs high) drives this split, validating that the scoring captures opposing vibe preferences effectively.

### Chill Lofi vs Adversarial (Sad High Energy)
Chill Lofi focuses on chill, low-energy songs with acoustic elements, while the adversarial profile ranks high-energy rock despite sad mood mismatches. This opposition in energy levels makes sense, as the system correctly separates relaxing tracks from intense ones, though mood rigidity limits nuance.

### Deep Intense Rock vs Adversarial (Sad High Energy)
Both are rock-focused with high energy, but Deep Intense Rock ranks intense-mood songs first, while the adversarial (sad mood) scores lower on mood but still tops with genre matches. The mood difference (intense vs sad) creates valid variety, but genre's dominance ensures rock songs persist, exposing the algorithm's categorical prioritization.

---

## Responsible AI Reflection

### Limitations and Biases in the System

The most significant bias is one the system cannot fix on its own: the knowledge base is written by one person with one cultural perspective on music. The 33 chunks describe genres like lofi, jazz, blues, and reggae in terms of how they tend to be experienced by Western listeners. A listener who grew up with reggaeton, Afrobeats, or K-pop will find their genres either absent or described from the outside. The system does not know what it does not know — it will confidently retrieve a chunk and generate a warm explanation even when the underlying knowledge is incomplete or culturally narrow.

The scoring engine carries forward a structural bias from Modules 1–3: genre is a hard categorical match worth a full point, which means a listener who listens across genres gets systematically worse recommendations than one who stays within a single genre. The RAG layer softens this in the explanation — it can write "this ambient track shares lofi's quiet warmth" — but it does not change the score. The rank order still punishes cross-genre taste.

The LLM (llama3.2:3b) introduces another limitation: it is a general-purpose model, not a music-specialized one. Given thin or generic retrieval context, it will produce explanations that sound plausible but are not specifically grounded. There is no mechanism to detect when the model has drifted from the retrieved knowledge into trained associations. The quality guard catches responses that are too short or too long, but it cannot detect confident-sounding responses that are factually off.

Finally, the system has no memory of past interactions. Every run is stateless. A listener who explicitly skips jazz songs every session will never have that preference learned or respected.

### Could the System Be Misused?

The most realistic misuse scenario is not dramatic — it is subtle. A system like this could be used to push content rather than recommend it. If the knowledge base chunks are written to frame certain songs or genres favorably regardless of the user's actual profile, the LLM will faithfully produce explanations that justify those songs. The user sees a warm, personalized-sounding explanation and has no way to know the retrieved "knowledge" was curated to lead them there. This is the same mechanism that makes RAG powerful for grounding responses that also makes it exploitable for laundering bias through confident prose.

A second misuse vector is scale. This project runs on a personal catalog of 18 songs. At scale — a real music platform with millions of tracks — the same architecture could be used to surface songs from labels or artists who pay for placement, with the RAG layer generating explanations that make commercial decisions look like personal curation.

Prevention at this project's level is straightforward: keep the knowledge base open and readable, document its contents and limitations, and do not give the system authority over anything consequential. At production scale, prevention requires auditing what goes into the knowledge corpus, labeling when recommendations are influenced by factors other than the user's stated preferences, and giving users meaningful control over how their taste profile is built and used.

### What Surprised Me During Reliability Testing

The most surprising result was test 10: that the retrieval query had to include both the user's genre and the song's genre to work correctly for off-genre recommendations. This only became clear during the planning critique phase — not during testing. By the time the tests ran, the architecture already accounted for it. What surprised me was how invisible the flaw would have been if I had just built the first draft and run it. The system would have still produced output. The lofi user would have still gotten explanations for the ambient song. They would have just been worse — more generic, less grounded in the actual relationship between the two genres — and without a test specifically checking the query construction, there would have been no signal that something was missing.

The confidence scoring also produced a result worth noting: the "Coffee Shop Stories" jazz/relaxed song for a lofi/chill user scored 0.68 retrieval confidence — the only MODERATE in the set. That song sits at the bottom of the recommendation list anyway (score-based), but the confidence score independently flags it as the weakest retrieval match. The two metrics are measuring different things and they agreed. That kind of convergent signal is more reassuring than either metric alone.

### Collaboration with AI During This Project

This project was built collaboratively with Claude (Anthropic). Claude contributed the architecture, wrote all the code, ran the reliability checks, and generated the knowledge base content. My role was to define the requirements, ask the clarifying questions, review the plan across three critique passes, and redirect when something was off — most importantly, correcting the wrong project folder early in the build.

**One instance where the AI gave a genuinely helpful suggestion:** During the planning critique phase, Claude caught that the initial retrieval query was built from the user profile alone. The specific suggestion was to build the query from both the user's genre and mood AND the song's genre and mood — so that for an off-genre recommendation like ambient surfacing in a lofi user's list, the system retrieves the lofi-ambient crossover chunk rather than just a generic lofi chunk. This was not a surface-level improvement; it changed what the retrieved knowledge contained and therefore what the LLM had to work with. Catching it before writing any embedding code saved real rework.

**One instance where the AI's suggestion was flawed:** The first version of the `knowledge_base.txt` file was written to the wrong project folder — the Module 1–3 starter project instead of project4. Claude wrote 33 chunks of carefully crafted music knowledge to a folder that had nothing to do with the actual submission. The file had to be copied manually after the mistake was caught. This is a straightforward failure of not verifying context before acting — the AI proceeded confidently with a plausible-looking path without confirming which project was active. The lesson is the same one the system's own fallback logic encodes: always check the output, not just the intent.
