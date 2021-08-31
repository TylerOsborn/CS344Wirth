from flask import Flask, render_template, redirect, url_for
app = Flask(__name__)


@app.route("/")
def home_page():
    return render_template("store/home_page.html")

if __name__ == "__main__":
    app.run(debug=True)
