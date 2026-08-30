from flask import Flask, render_template, request

from recommendation import recommend, recommend_tamil


app = Flask(__name__)


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

        if not recommendations:

            message = (
                "Movie not found. "
                "Please enter a valid movie name."
            )

    return render_template(
        "index.html",
        recommendations=recommendations,
        movie_name=movie_name,
        language=language,
        message=message
    )


if __name__ == "__main__":
    app.run(debug=True)