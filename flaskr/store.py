from flask import Flask, render_template, redirect, url_for, request
import pandas as pd
import test_recommender as recommender
import os
import json

app = Flask(__name__)


@app.route("/")
def home_page():
    return render_template("store/home_page.html")


@app.route("/cluster_visual")
def cluster_page():
    return render_template("store/cluster_page.html")


@app.route("/model_details")
def model_page():
    return render_template("store/model_page.html")


@app.route("/product_search")
def products_page():
    return render_template("store/products_page.html", data=None, data_len=0)


@app.route("/item", methods=["GET", "POST"])
def item_page():
    if request.method == "GET":
        product_id = request.args["product_id"]
        recommendations, scores = recommender.get_recommendations(product_id)
        product = recommender.get_item(product_id)
        clusters = [recommender.get_cluster(product_id)]
        silhouette = recommender.get_silhouette()
        words = [recommender.get_word_similarities(product_id, i) for i in recommendations['catalogItemId']]
        for i in range(len(recommendations)):
            clusters.append(recommender.get_cluster(recommendations['catalogItemId'].iloc[i]))
        return render_template("store/item_page.html", product=product, recommendations=recommendations, scores=scores,
                               clusters=clusters, silhouette=silhouette, words=words)
    return render_template("store/item_page.html", product=None, recommendations=None, clusters=None, silhouette=None)


@app.route("/search", methods=["GET", "POST"])
def search():
    query = request.args["query"]
    data = recommender.search_products(query)
    return render_template("store/products_page.html", data=data, data_len=len(data))


@app.route("/product", methods=["GET", "POST"])
def get_item_page():
    query = request.args["query"]
    recommendations = recommender.get_recommendations(query, recommender.cosine_sim)
    return "<h1> hello </h1>"


@app.route('/visual_data', methods=['GET'])
def cluster_data():
    f = open('cluster_data.json', )
    data = json.load(f)
    f.close()
    return json.dumps(data)


if __name__ == "__main__":
    app.run(debug=True)

