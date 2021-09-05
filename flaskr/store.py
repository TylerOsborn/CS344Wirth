from flask import Flask, render_template, redirect, url_for
import pandas as pd
import recommendation as recommendation

app = Flask(__name__)


@app.route("/")
def home_page():
    return render_template("store/home_page.html")

@app.route("/product_search")
def products_page():
    return render_template("store/products_page.html", data=recommendation.products)

@app.route("/item")
def item_page():
    return render_template("store/item_page.html")

if __name__ == "__main__":
    app.run(debug=True)
