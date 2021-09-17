from flask import Flask, render_template, redirect, url_for, request
import pandas as pd
import test_recommender as recommendation

app = Flask(__name__)


@app.route("/")
def home_page():
    return render_template("store/home_page.html")

@app.route("/product_search")
def products_page():
    return render_template("store/products_page.html", data=None, data_len=0)

@app.route("/item")
def item_page():
    return render_template("store/item_page.html")

@app.route("/search", methods=["GET", "POST"])
def search():
    query = request.args['query']
    print(query)
    data = recommendation.search_products(query)
    return render_template("store/products_page.html", data=data, data_len=len(data))
if __name__ == "__main__":
    app.run(debug=True)
