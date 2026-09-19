import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# MALAYALAM MOVIE RECOMMENDATION
# ============================================================

mal = pd.read_csv("malayalam_movies.csv")

# Remove rows with missing values in required columns
mal = mal.dropna(
    subset=[
        "title",
        "genres",
        "keywords",
        "cast",
        "directors",
        "overview"
    ]
).reset_index(drop=True)


# Create tags
mal["tags"] = (
    mal["genres"].astype(str) + " " +
    mal["keywords"].astype(str) + " " +
    mal["cast"].astype(str) + " " +
    mal["directors"].astype(str) + " " +
    mal["overview"].astype(str)
)


# Malayalam CountVectorizer
cv = CountVectorizer(
    max_features=5000,
    stop_words="english"
)


# Convert movie tags into vectors
movie_vectors = cv.fit_transform(mal["tags"]).toarray()


# Calculate cosine similarity
similarity = cosine_similarity(movie_vectors)


def recommend(movie_name):

    # Clean entered movie name
    movie_name = movie_name.strip().lower()

    # Convert titles to lowercase
    movie_titles = (
        mal["title"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Check whether movie exists
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

    # Get top 5 similar movies
    for index, score in similarity_scores:

        # Skip the movie entered by the user
        if index == movie_index:
            continue

        movie = mal.iloc[index]

        recommendations.append({
            "title": movie["title"],
            "genre": movie["genres"],
            "rating": movie["rating"],
            "similarity": round(score * 100, 2),
            "poster": movie.get("poster_url", "")
        })

        if len(recommendations) == 5:
            break

    return recommendations


# ============================================================
# TAMIL MOVIE RECOMMENDATION
# ============================================================

tam = pd.read_csv("tamil_movies.csv")

# Replace missing values
tam = tam.fillna("")


# Convert required columns to strings
tam["Genre"] = tam["Genre"].astype(str)
tam["Director"] = tam["Director"].astype(str)
tam["Actor"] = tam["Actor"].astype(str)


# Create Tamil movie tags
tamil_tags = (
    tam["Genre"] + " " +
    tam["Director"] + " " +
    tam["Actor"]
)


# Make sure all tags are strings
tamil_tags = tamil_tags.astype(str)


# Tamil CountVectorizer
cv_tamil = CountVectorizer(
    input="content",
    max_features=5000,
    stop_words="english"
)


# Convert Tamil movie tags into vectors
tam_vectors = cv_tamil.fit_transform(
    tamil_tags
)


# Calculate Tamil cosine similarity
tamil_similarity = cosine_similarity(
    tam_vectors
)


def recommend_tamil(movie_name):

    # Clean entered movie name
    movie_name = movie_name.strip().lower()

    # Convert Tamil movie names to lowercase
    movie_titles = (
        tam["MovieName"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Check whether movie exists
    if movie_name not in movie_titles.values:
        return []

    # Find movie index
    movie_index = movie_titles[
        movie_titles == movie_name
    ].index[0]

    # Get similarity scores
    similarity_scores = list(
        enumerate(tamil_similarity[movie_index])
    )

    # Sort by similarity
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    # Get top 5 similar movies
    for index, score in similarity_scores:

        # Skip the selected movie
        if index == movie_index:
            continue

        movie = tam.iloc[index]

        # Select rating
        rating = movie["Rating"]

        # If Rating is empty, use movie_rating
        if str(rating).strip() == "":
            rating = movie["movie_rating"]

        recommendations.append({
            "title": movie["MovieName"],
            "genre": movie["Genre"],
            "rating": rating,
            "similarity": round(score * 100, 2),
            "poster": ""
        })

        if len(recommendations) == 5:
            break

    return recommendations