import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# import tensorflow.keras as tf

data = pd.read_json('SampleData.json')
item_data = pd.DataFrame([i['_source'] for i in data["hits"]["hits"]])


tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(item_data['name'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(item_data.index, index=item_data['catalogItemId']).drop_duplicates()

print(indices)

def get_recommendations(product_id, cosine_sim=cosine_sim):
    idx = indices[product_id]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4]
    item_indices = [i[0] for i in sim_scores]

    return item_data["name"].iloc[item_indices]

print(get_recommendations('WTD5401CHXL'))