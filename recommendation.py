import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# MALAYALAM MOVIE RECOMMENDATION
# =========================================================

# Load Malayalam dataset
mal = pd.read_csv("malayalam_movies.csv")

# Remove missing values from required columns
mal = mal.dropna(
    subset=["title", "genres", "keywords", "cast", "directors", "overview"]
).reset_index(drop=True)


# Create Tags
mal["tags"] = (
    mal["genres"].astype(str) + " " +
    mal["keywords"].astype(str) + " " +
    mal["cast"].astype(str) + " " +
    mal["directors"].astype(str) + " " +
    mal["overview"].astype(str)
)


# Convert text into numerical vectors
cv = CountVectorizer(
    max_features=5000,
    stop_words="english"
)

movie_vectors = cv.fit_transform(mal["tags"]).toarray()


# Calculate Cosine Similarity
similarity = cosine_similarity(movie_vectors)


# Malayalam Recommendation Function
def recommend(movie_name):

    movie_name = movie_name.strip().lower()

    movie_titles = (
        mal["title"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Check movie
    if movie_name not in movie_titles.values:
        return []

    # Find movie index
    movie_index = movie_titles[
        movie_titles == movie_name
    ].index[0]

    # Get similarity scores
    similarity_scores = list(
        enumerate(similarity[movie_index])
    )

    # Sort by similarity
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for i, score in similarity_scores:

        # Skip selected movie
        if i == movie_index:
            continue

        movie = mal.iloc[i]

        # Get rating safely
        rating = movie["rating"] if "rating" in mal.columns else "N/A"

        # Get genre safely
        genre = movie["genres"] if "genres" in mal.columns else "N/A"

        # Get poster safely
        poster = (
            movie["poster_url"]
            if "poster_url" in mal.columns
            else ""
        )

        recommendations.append({
            "title": movie["title"],
            "genre": genre,
            "rating": rating,
            "similarity": round(score * 100, 1),
            "poster": poster
        })

        if len(recommendations) == 5:
            break

    return recommendations


# =========================================================
# TAMIL MOVIE RECOMMENDATION
# =========================================================

# Load Tamil dataset
tam = pd.read_csv("tamil_movies.csv")


# Create Tags
tam["tags"] = (
    tam["Genre"].astype(str) + " " +
    tam["Director"].astype(str) + " " +
    tam["Actor"].astype(str)
)


# Convert Tamil movie tags into numerical vectors
cv_tamil = CountVectorizer(
    stop_words="english"
)

tamil_vectors = cv_tamil.fit_transform(
    tam["tags"]
).toarray()


# Calculate Tamil Cosine Similarity
tamil_similarity = cosine_similarity(
    tamil_vectors
)


# Tamil Recommendation Function
def recommend_tamil(movie_name):

    movie_name = movie_name.strip().lower()

    movie_titles = (
        tam["MovieName"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Check movie
    if movie_name not in movie_titles.values:
        return []

    # Find movie index
    movie_index = movie_titles[
        movie_titles == movie_name
    ].index[0]

    # Get similarity scores
    similarity_scores = list(
        enumerate(
            tamil_similarity[movie_index]
        )
    )

    # Sort by similarity
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for i, score in similarity_scores:

        # Skip selected movie
        if i == movie_index:
            continue

        movie = tam.iloc[i]

        # Get rating
        if "Rating" in tam.columns:
            rating = movie["Rating"]
        elif "movie_rating" in tam.columns:
            rating = movie["movie_rating"]
        else:
            rating = "N/A"

        recommendations.append({
            "title": movie["MovieName"],
            "genre": movie["Genre"],
            "rating": rating,
            "similarity": round(score * 100, 1),
            "poster": ""
        })

        if len(recommendations) == 5:
            break

    return recommendations