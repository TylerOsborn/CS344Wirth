import json
import csv
import pandas as pd
import math
import nltk
from rake_nltk import Rake
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
#nltk.download('stopwords')
#nltk.download('punkt')
pd.set_option('display.max_columns', 100)
df = pd.read_csv('data_file.csv')

df.head()


def create_csv_file():
    with open('Electronics.json') as json_file:
        elec_data = json.load(json_file)

    with open('AnimalsAndPetSupplies.json') as json_file:
        animal_data = json.load(json_file)

    with open('BusinessAndIndustrial.json') as json_file:
        business_data = json.load(json_file)

    with open('Hardware.json') as json_file:
        hardware_data = json.load(json_file)

    with open('HomeAndGarden.json') as json_file:
        garden_data = json.load(json_file)

    data = elec_data + animal_data + business_data + hardware_data + garden_data

    # Open a csv file for writing
    data_file = open('data_file.csv', 'w')

    # Create the csv writer object
    csv_writer = csv.writer(data_file)

    # Counter variable used for writing
    # headers to the CSV file
    count = 0
    category_flag = True

    header = ["name", "description", "manufacturer", "category", "labels", "price", "color", "size", "adult", "gender", "productWeight", "weightUnit", "descriptionTags", "nameTags", "tags"]

    for product in data:
        tag_list = ''
        product = product["_source"]
        if count == 0:
            # Write the headers
            csv_writer.writerow(header)
            count += 1

        name = product["name"].lower()
        r = Rake()
        r.extract_keywords_from_text(name)
        tags = r.get_word_degrees()
        name_tags = list(tags.keys())
        tag_list = tag_list + ' '.join(name_tags) + ' '

        try:
            description = product["description"].lower()
            r = Rake()
            r.extract_keywords_from_text(description)
            tags = r.get_word_degrees()
            description_tags = list(tags.keys())
            tag_list = tag_list + ' '.join(description_tags) + ' '
        except KeyError:
            description = "N/A"
            description_tags = []

        category = "N/A"

        try:
            category = product["category"]
            category = category.split(">")
            for i in range(0, len(category)):
                category[i] = category[i].strip().lower()
            tag_list = tag_list + ' '.join(category) + ' '

        except KeyError:
            category_flag = False

        # Object
        try:
            try:
                labels = product["labels"]
                for i in range(0, len(labels)):
                    labels[i] = labels[i].strip().lower()
                tag_list = tag_list + ' '.join(labels) + ' '
            except IndexError:
                labels = "N/A"
        except KeyError:
            labels = "N/A"

        try:
            manufacturer = product["manufacturer"]
            tag_list = tag_list + manufacturer + ' '
        except KeyError:
            manufacturer = "N/A"

        price = product["currentPrice"]

        try:
            # Object
            color = product["physicalAttributes"]["color"]
        except KeyError:
            color = "N/A"

        try:
            size = product["physicalAttributes"]["size"]
        except KeyError:
            size = "N/A"

        try:
            adult = product["targetDemographics"]["adult"]
        except KeyError:
            adult = "N/A"

        try:
            weight = product["physicalAttributes"]["productWeight"]
        except KeyError:
            weight = "N/A"

        try:
            weightUnit = product["physicalAttributes"]["weightUnit"]
        except KeyError:
            weightUnit = "N/A"

        try:
            gender = product["targetDemographics"]["gender"].lower()
        except KeyError:
            gender = "N/A"

        if category_flag is False:
            category = product["taxonomy"]["category"]

        # Writing data of CSV file
        row = [name.lower(), description.lower(), manufacturer.lower(), category, labels, price,
               color, size, adult, gender.lower(), weight, weightUnit, description_tags, name_tags, tag_list]
        csv_writer.writerow(row)

    data_file.close()


df = df[['name', 'tags']]

# Generating the count matrix
count = CountVectorizer()
count_matrix = count.fit_transform(df['tags'])

# generating the cosine similarity matrix
cosine_sim = cosine_similarity(count_matrix, count_matrix)


def recommendation(query, cosine_sim):
    products = []
    idx = 4
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)
    top_results = list(score_series.iloc[1:20].index)

    for i in top_results:
        products.append(list(df.index)[i])

    return products


query = "garmin approach g3 touchscreen gps"
recommendation(query, cosine_sim)

recommendation_indexes = recommendation(query, cosine_sim)
print(recommendation_indexes)
