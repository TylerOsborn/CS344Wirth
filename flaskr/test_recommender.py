import numpy as np
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import KMeans

pd.set_option('display.max_colwidth', None)

data = pd.read_json('../combined.json')
item_data = pd.DataFrame([i['_source'] for i in data['data']])


def homogenize(data):
    for i in data:
        bool_series = pd.isnull(data[i])
        data.loc[bool_series, i] = ""


homogenize(item_data)


# combine all relevant datafields into a single field for processing
def soupify(data):
    data['soup'] = ""
    for i in data:
        data['soup'] += str(data[i]) + ' '


# soupify(item_data)
item_data['soup'] = item_data['description'].str.lower() + ', ' + item_data['name'].str.lower()

try:
    item_data['soup'] += ',' + item_data['category'].str.lower()
except KeyError | ValueError:
    print("Null value")

try:
    item_data['soup'] += ',' + item_data['manufacturer'].str.lower()
except KeyError | ValueError:
    print("Null value")


def get_recommender(data):
    # object to remove all non-necessary words from the description
    tfidf = TfidfVectorizer(stop_words='english')
    # matrix of keywords found in description (tfidf now has a list of all descriptor words)
    tfidf_matrix = tfidf.fit_transform(item_data['soup'])
    # cosine similarity matrix of all items
    matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
    #penalise by price ratio
    for i in range(len(matrix)):
        for j in range(i,len(matrix[i])):
            p1 = data['dollarPrice'].iloc[i]
            p2 = data['dollarPrice'].iloc[j]
            r = min([p1,p2])/max([p1,p2])
            matrix[i][j] = matrix[i][j]*r
            matrix[j][i] = matrix[i][j]
    return matrix


cosine_sim = get_recommender(item_data)


def search_products(query, data=item_data):
    results = data[data['soup'].str.find(query.lower()) > 0]
    return results


def get_item(query, data=item_data, column="catalogItemId"):
    product = data[data[column] == query]
    if not product.empty:
        return product
    return None


def get_recommendations(index_id, data=item_data, return_num=6, index_name='catalogItemId'):
    cosine_sim = get_recommender(data)
    indices = pd.Series(data.index, index=data[index_name]).drop_duplicates()

    idx = indices[index_id]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1: return_num + 1]
    item_indices = [i[0] for i in sim_scores if 0.11 <= i[1] <= 0.98]
    scores = [i[1] for i in sim_scores if 0.11 <= i[1] <= 0.98]

    return pd.DataFrame(item_data.iloc[item_indices]), scores


tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(item_data['soup'])


def get_cluster(product_id):
    product = get_item(product_id)
    if product is not None:
        kmeans = KMeans(n_clusters=39, random_state=1).fit(tfidf_matrix)
        cluster = kmeans.predict(tfidf.transform([product['name'].iloc[0].lower() + "," +
                                                  product['description'].iloc[0].lower() +
                                                  product['category'].iloc[0].lower()]))
        return cluster[0]
    else:
        return -1

# print(get_recommendations('022150254', item_data, return_num=10, index_name='catalogItemId'))
# search_products("gps")
# get_item("022150254")
