import pandas as pd

data = pd.read_json('SampleData.json')
products = [i["_source"] for i in data["hits"]["hits"]]
products = pd.DataFrame(products)

def remove_brand_names(data):
    data['name'] = data.apply(lambda row: row['name'].replace(row['manufacturer'], ''), axis=1)

def remove_size_from_name(data):
    data['name'] = data.apply(lambda row: row['name'].replace(" - Size: ", ''), axis=1)
    data['name'] = data.apply(lambda row: row['name'].replace(row['size'], ''), axis=1)

def remove_genders(data):
    gender_words = ["Women\'s", "Men\'s", "Men", "Male", "Female", "Women"]
    for i in gender_words:
        data['name'] = data.apply(lambda row: row['name'].replace(i, ''), axis=1)

remove_genders(products)
remove_brand_names(products)
remove_size_from_name(products)

print(products.loc[0]['name'].__len__)
