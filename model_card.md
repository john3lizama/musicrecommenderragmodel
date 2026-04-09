# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

Give your model a short, descriptive name.  
Example: **VibeFinder 1.0**  

---

## 2. Intended Use  

Describe what your recommender is designed to do and who it is for. 

Prompts:  

- What kind of recommendations does it generate  
- What assumptions does it make about the user  
- Is this for real users or classroom exploration  

---

## 3. How the Model Works  

Explain your scoring approach in simple language.  

Prompts:  

- What features of each song are used (genre, energy, mood, etc.)  
- What user preferences are considered  
- How does the model turn those into a score  
- What changes did you make from the starter logic  

Avoid code here. Pretend you are explaining the idea to a friend who does not program.

---

## 4. Data  

Describe the dataset the model uses.  

Prompts:  

- How many songs are in the catalog  
- What genres or moods are represented  
- Did you add or remove data  
- Are there parts of musical taste missing in the dataset  

---

## 5. Strengths  

Where does your system seem to work well  

Prompts:  

- User types for which it gives reasonable results  
- Any patterns you think your scoring captures correctly  
- Cases where the recommendations matched your intuition  

---

## 6. Limitations and Bias 

Where the system struggles or behaves unfairly. 

Prompts:  

- Features it does not consider  
- Genres or moods that are underrepresented  
- Cases where the system overfits to one preference  
- Ways the scoring might unintentionally favor some users  

The system over-prioritizes genre matches due to the high weight assigned to genre (1.0 points), which can overshadow other musical attributes like energy or mood, especially in a small dataset with uneven genre distribution. For instance, users preferring rare genres like reggae or classical receive limited variety since only one song exists per genre, forcing the system to rely heavily on that single match. This creates filter bubbles where users with common preferences (e.g., pop or lofi) get consistent, high-scoring recommendations, while those with niche or extreme tastes see repetitive results. Experiments showed that halving the genre weight improved energy sensitivity but didn't fully eliminate this bias, highlighting the need for a larger, more balanced dataset to prevent overfitting to categorical labels.

---

## 7. Evaluation  

How you checked whether the recommender behaved as expected. 

I tested four distinct user profiles: High-Energy Pop (high energy, happy mood, pop genre), Chill Lofi (low energy, chill mood, lofi genre), Deep Intense Rock (high energy, intense mood, rock genre), and an adversarial profile (high energy, sad mood, rock genre) to stress-test conflicting preferences. I looked for relevance to stated preferences, ranking consistency, and how well the system handled edge cases like mood mismatches. What surprised me was how the adversarial profile still ranked rock songs highly due to genre dominance, despite the sad mood mismatch, revealing the system's bias toward categorical features over nuanced conflicts. I also ran a weight shift experiment (doubling energy, halving genre), which increased sensitivity to energy but kept rankings largely stable, showing the algorithm's robustness but also its limitations in small datasets.

---

## 8. Future Work  

Ideas for how you would improve the model next.  

Prompts:  

- Additional features or preferences  
- Better ways to explain recommendations  
- Improving diversity among the top results  
- Handling more complex user tastes  

---

## 9. Personal Reflection  

A few sentences about your experience.  

Prompts:  

- What you learned about recommender systems  
- Something unexpected or interesting you discovered  
- How this changed the way you think about music recommendation apps  
