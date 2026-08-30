import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Malayalam movie dataset
mal = pd.read_csv("malayalam_movies.csv")

# Remove missing values
mal = mal.dropna()

# Create Tags column
mal["tags"] = (
    mal["genres"] + " " +
    mal["keywords"] + " " +
    mal["cast"] + " " +
    mal["directors"] + " " +
    mal["overview"]
)

# Convert text into numerical vectors
cv = CountVectorizer(max_features=5000, stop_words="english")
movie_vectors = cv.fit_transform(mal["tags"]).toarray()

print("Movie Vector Shape:", movie_vectors.shape)

# Calculate Cosine Similarity
similarity = cosine_similarity(movie_vectors)

print("Similarity Matrix Shape:", similarity.shape)

print("\nCosine Similarity Calculation Completed Successfully")

