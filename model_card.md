# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

TuneMatcher 1.0  

---

## 2. Intended Use  

Suggests songs that match user preferences for genre, mood, and energy.  
Designed for classroom fun and learning about recommenders.  
Not for real music apps or serious recommendations.  

---

## 3. How the Model Works  

Scores songs by matching genre and mood exactly.  
Gives points based on how close energy levels are.  
Adds smaller points for other features like happiness or acoustic sound.  

---

## 4. Data  

Uses 18 songs from a CSV file.  
Songs have genre, mood, energy, tempo, and other features.  
Covers pop, lofi, rock, and more genres.  
Small dataset misses many music types.  

---

## 5. Strengths  

Works well for common preferences like pop or lofi.  
Captures basic matches in genre and energy.  
Simple scoring is easy to understand.  

---

## 6. Limitations and Bias 

Favors genre matches too much, ignoring other features.  
Small dataset limits variety for rare genres.  
May create bubbles where users get similar songs.  

---

## 7. Evaluation  

Tested with four user profiles: pop fan, chill lover, rock enthusiast, and mixed preferences.  
Checked if top songs matched stated likes.  
Ran experiments by changing scoring weights.  

---

## 8. Future Work  

Add more songs to the dataset.  
Balance scoring weights better.  
Include user feedback for better matches.  

---

## 9. Personal Reflection  

My biggest learning moment was realizing how data biases can sneak into simple systems. The genre matching dominated everything, even when it didn't make sense, showing me why fair AI matters.  

AI tools like Copilot helped a lot with writing code and explaining concepts quickly. But I had to double-check the scoring logic myself because the AI sometimes suggested weights that didn't match my testing results.  

I was surprised that such a basic algorithm still felt like real recommendations. Even with just matching and proximity scores, the top songs often matched what I'd expect for a user profile.  

If I extended this, I'd try adding collaborative filtering using user ratings, or machine learning to learn weights automatically from data. I'd also build a web interface for real users to test it.  
