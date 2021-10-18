import numpy as np
import pandas as pd
import re
from sklearn.metrics import silhouette_samples, silhouette_score

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import KMeans

pd.set_option("display.max_colwidth", None)

data = pd.read_json("../combined.json")
item_data = pd.DataFrame([i["_source"] for i in data["data"]])


def homogenize(data):
    for i in data:
        bool_series = pd.isnull(data[i])
        data.loc[bool_series, i] = ""


homogenize(item_data)


# combine all relevant datafields into a single field for processing
def soupify(data):
    data["soup"] = ""
    for i in data:
        data["soup"] += str(data[i]) + " "


# soupify(item_data)
item_data["soup"] = (
    item_data["description"].str.lower() + ", " + item_data["name"].str.lower()
)

try:
    item_data["soup"] += "," + item_data["category"].str.lower()
except KeyError | ValueError:
    print("Null value")

try:
    item_data["soup"] += "," + item_data["manufacturer"].str.lower()
except KeyError | ValueError:
    print("Null value")


def get_recommender(data):
    # object to remove all non-necessary words from the description
    tfidf = TfidfVectorizer(stop_words="english")
    # matrix of keywords found in description (tfidf now has a list of all descriptor words)
    tfidf_matrix = tfidf.fit_transform(item_data["soup"])
    # cosine similarity matrix of all items
    matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
    # penalise by price ratio
    for i in range(len(matrix)):
        for j in range(i, len(matrix[i])):
            p1 = data["dollarPrice"].iloc[i]
            p2 = data["dollarPrice"].iloc[j]
            r = min([p1, p2]) / max([p1, p2])
            matrix[i][j] = matrix[i][j] * r
            matrix[j][i] = matrix[i][j]
    return matrix


cosine_sim = get_recommender(item_data)


def search_products(query, data=item_data):
    results = data[data["soup"].str.find(query.lower()) > 0]
    return results


def get_item(query, data=item_data, column="catalogItemId"):
    product = data[data[column] == query]
    if not product.empty:
        return product
    return None


def get_recommendations(
    index_id, data=item_data, return_num=6, index_name="catalogItemId"
):
    # cosine_sim = get_recommender(data)
    indices = pd.Series(data.index, index=data[index_name]).drop_duplicates()

    idx = indices[index_id]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    count = 1
    original_item = data.iloc[idx]
    item_group = []
    return_items = []
    if "variant" in original_item:
        if "itemGroupId" in original_item["variant"]:
            item_group = [str(original_item["variant"]["itemGroupId"])]

    while len(return_items) < return_num and count < len(sim_scores):
        i = sim_scores[count][0]
        item = data.iloc[i]
        if "variant" in item:
            if "itemGroupId" in item["variant"]:
                var = str(item["variant"]["itemGroupId"])
                if not var in item_group:
                    return_items.append(sim_scores[count])
                    item_group.append(var)
            else:
                return_items.append(sim_scores[count])
        else:
            return_items.append(sim_scores[count])
        count += 1
    item_indices = [i[0] for i in return_items]
    scores = [i[1] for i in return_items]

    return pd.DataFrame(item_data.iloc[item_indices]), scores


tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(item_data["soup"])


# get the first ten words used to predict similarity of products
def get_word_similarities(id1, id2):
    results = []
    item1 = get_item(id1)
    item2 = get_item(id2)
    features = tfidf.get_feature_names_out()
    soup1 = str(item1["soup"]).split(" ")
    soup2 = str(item2["soup"]).split(" ")
    count = 10
    for i in soup1:
        if count <= 0:
            break
        if i in soup2 and i in features and not i in results:
            results.append(i)
            count = count - 1
    return results


def get_cluster(product_id):
    product = get_item(product_id)
    if product is not None:
        kmeans = KMeans(n_clusters=30, random_state=1).fit(tfidf_matrix)
        cluster = kmeans.predict(
            tfidf.transform(
                [
                    product["name"].iloc[0].lower()
                    + ","
                    + product["description"].iloc[0].lower()
                    + product["category"].iloc[0].lower()
                ]
            )
        )
        return cluster[0]
    else:
        return -1


def get_silhouette():
    k = 30
    kmeans = KMeans(n_clusters=k, random_state=1)
    cluster_labels = kmeans.fit_predict(tfidf_matrix.toarray())
    score = silhouette_score(tfidf_matrix.toarray(), cluster_labels, random_state=1)
    return score


# search_products("gps")
# get_item("022150254")
