import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Get the absolute path of movies.csv
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "movies.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "similarity.pkl")

# Debugging: Print the exact path being used
print(f"🔍 Looking for movies.csv at: {DATA_PATH}")

# Check if movies.csv exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ movies.csv NOT found at {DATA_PATH}. Check file location.")

# Load dataset
movies = pd.read_csv(DATA_PATH)
movies.fillna("", inplace=True)
movies.fillna("", inplace=True)  # Handle missing values
movies["content"] = (
    movies["title"] + " " + 
    movies["genres"] + " " + 
    movies["overview"] + " " + 
    movies["keywords"] + " " + 
    movies["cast"] + " " + 
    movies["director"]
)

# ✅ Convert text data to feature vectors
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10_000,  # Focus on the most important words
    ngram_range=(1, 2)  # Consider single words and bigrams (word pairs)
)


movie_vectors = vectorizer.fit_transform(movies["content"])

# ✅ Compute similarity matrix
from sklearn.metrics.pairwise import linear_kernel

similarity = linear_kernel(movie_vectors, movie_vectors)  # More effective for sparse data


# ✅ Ensure 'models' directory exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# ✅ Save similarity matrix
with open(MODEL_PATH, "wb") as file:
    pickle.dump(similarity, file)
print("✅ Similarity matrix saved successfully!")

# ✅ Load precomputed similarity matrix
with open(MODEL_PATH, "rb") as file:
    similarity = pickle.load(file)
print("✅ Similarity matrix loaded successfully!")

# 🎯 Recommendation function
def recommend_movies(movie_title, num_recommendations=5):
    """Returns top recommended movies based on cosine similarity."""
    if movie_title not in movies["title"].values:
        return ["❌ Movie not found. Try another title."]
    
    # Get index of the movie
    movie_idx = movies[movies["title"] == movie_title].index[0]
    
    # Get similarity scores & sort
    scores = list(enumerate(similarity[movie_idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    
    # Get recommended movie titles
    recommended_movies = [movies.iloc[i[0]]["title"] for i in sorted_scores]
    return recommended_movies
if __name__ == "__main__":
    movie_name = input("Enter a movie name: ")
    recommendations = recommend_movies(movie_name)
    
    print("\n🎬 Recommended Movies:")
    for idx, rec in enumerate(recommendations, start=1):
        print(f"{idx}. {rec}")
