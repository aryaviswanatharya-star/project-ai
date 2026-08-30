import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

# Load datasets
mal = pd.read_csv("malayalam_movies.csv")
tam = pd.read_csv("tamil_movies.csv")

# Remove missing values
mal = mal.dropna()
tam = tam.dropna()

# Create Tags Column
mal["tags"] = (
    mal["genres"] + " " +
    mal["keywords"] + " " +
    mal["cast"] + " " +
    mal["directors"] + " " +
    mal["overview"]
)

# Text Vectorization
cv = CountVectorizer(max_features=5000, stop_words="english")
movie_vectors = cv.fit_transform(mal["tags"]).toarray()

print("Movie Vector Shape:", movie_vectors.shape)

# Feature Scaling
scaler = StandardScaler()

mal[["rating", "vote_count", "popularity", "runtime"]] = scaler.fit_transform(
    mal[["rating", "vote_count", "popularity", "runtime"]]
)

tam[["Rating", "PeopleVote", "Hero_Rating", "movie_rating"]] = scaler.fit_transform(
    tam[["Rating", "PeopleVote", "Hero_Rating", "movie_rating"]]
)

print("\nFeature Scaling Completed Successfully")

print("\nMalayalam Dataset")
print(mal[["rating", "vote_count", "popularity", "runtime"]].head())

print("\nTamil Dataset")
print(tam[["Rating", "PeopleVote", "Hero_Rating", "movie_rating"]].head())