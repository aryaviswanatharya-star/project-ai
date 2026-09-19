from flask import Flask, render_template, request
from recommendation import recommend, recommend_tamil
import pandas as pd

app = Flask(__name__)


# Load movie names for suggestions
mal = pd.read_csv("malayalam_movies.csv").fillna("")
tam = pd.read_csv("tamil_movies.csv").fillna("")

malayalam_movies = (
    mal["title"]
    .astype(str)
    .str.strip()
    .drop_duplicates()
    .tolist()
)

tamil_movies = (
    tam["MovieName"]
    .astype(str)
    .str.strip()
    .drop_duplicates()
    .tolist()
)


@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    movie_name = ""
    language = ""
    message = ""

    if request.method == "POST":
        language = request.form.get("language", "")
        movie_name = request.form.get("movie_name", "")

        if language == "Malayalam":
            recommendations = recommend(movie_name)

        elif language == "Tamil":
            recommendations = recommend_tamil(movie_name)

        else:
            message = "Please select a valid language."

        if language in ["Malayalam", "Tamil"] and not recommendations:
            message = "Movie not found. Please select a movie from the suggestions."

    return render_template(
        "index.html",
        recommendations=recommendations,
        movie_name=movie_name,
        language=language,
        message=message,
        malayalam_movies=malayalam_movies,
        tamil_movies=tamil_movies
    )


if __name__ == "__main__":
    app.run(debug=True)