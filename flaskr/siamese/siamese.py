import os
import numpy
import pandas
import re
import nltk
import pickle
import joblib
import tensorflow as tf
import keras
import fasttext

import keras.backend as K
from keras.preprocessing.text import Tokenizer, tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Concatenate, Conv2D, Flatten, Dense, Embedding, LSTM, Multiply, Dropout, Subtract, GlobalMaxPool2D, Add
from keras.layers.core import Lambda
from keras.models import Model
from keras.metrics import AUC

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from itertools import combinations, product
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from tqdm import tqdm

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

stop_words = set(stopwords.words('english')) 
stemmer = SnowballStemmer('english')
t = Tokenizer()
model = Model()

def stripunc(data): 
    return re.sub('[^A-Za-z]+', ' ', str(data), flags=re.MULTILINE|re.DOTALL)

def decontracted(phrase):
    # specific
    phrase = re.sub('won\'t', 'will not', phrase)
    phrase = re.sub('can\'t', 'can not', phrase)

    # general
    phrase = re.sub('n\'t', ' not', phrase)
    phrase = re.sub('\'re', ' are', phrase)
    phrase = re.sub('\'s', ' is', phrase)
    phrase = re.sub('\'d', ' would', phrase)
    phrase = re.sub('\'ll', ' will', phrase)
    phrase = re.sub('\'t', ' not', phrase)
    phrase = re.sub('\'ve', ' have', phrase)
    phrase = re.sub('\'m', ' am', phrase)
    return phrase

def compute(sent): 
    
    sent = decontracted(sent)
    sent = stripunc(sent) 
    
    words=word_tokenize(str(sent.lower())) 
    
    #Removing all single letter and and stopwords from question 
    sent1=' '.join(str(stemmer.stem(j)) for j in words if j not in stop_words and (len(j)!=1)) 
    sent2=' '.join(str(j) for j in words if j not in stop_words and (len(j)!=1)) 
    return sent1, sent2

def cosine_distance(vests):
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)

def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0],1)

def create_dataframe():
    if os.path.exists('df.pickle'):
        with open('df.pickle', 'rb') as f:
            df = pickle.load(f)
        return df

    df = pandas.read_json('combined.json')
    df = pandas.DataFrame([i['_source'] for i in df['data']])

    df['soup'] = df['name'] + df['description']

    l = list(product(df['soup'], df['soup']))

    df = pandas.DataFrame(data=list(product(df['soup'], df['soup'])), columns=['soup1','soup2']) 
    df['is_duplicate'] = numpy.where(df['soup1'] == df['soup2'],1,0)

    df.fillna(value = ' ',inplace = True)

    clean_stemmed_s1 = []
    clean_stemmed_s2 = []
    clean_s1 = []
    clean_s2 = []
    combined_stemmed_text = []

    for _, row in tqdm(df.iterrows()):
        css1, cs1 = compute(row['soup1'])
        css2, cs2 = compute(row['soup2'])
        clean_stemmed_s1.append(css1)
        clean_s1.append(cs1)
        clean_stemmed_s2.append(css2)
        clean_s2.append(cs2)
        combined_stemmed_text.append(css1+' '+css2)

    df['clean_stemmed_s1'] = clean_stemmed_s1
    df['clean_stemmed_s2'] = clean_stemmed_s2
    df['clean_s1'] = clean_s1
    df['clean_s2'] = clean_s2
    df['combined_stemmed_text'] = combined_stemmed_text
    with open('df.pickle', 'wb') as f:
        pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)

    return df

def create_model():

    fast = fasttext.train_unsupervised('descriptions.txt')

    df = create_dataframe()

    X_temp, X_test, y_temp, y_test = train_test_split(df[['clean_s1', 'clean_s2']], df['is_duplicate'], test_size=0.2, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
    X_train['text'] = X_train[['clean_s1','clean_s2']].apply(lambda x:str(x[0])+' '+str(x[1]), axis=1)

    t.fit_on_texts(X_train['text'].values)
    X_train['clean_s1'] = str(X_train['clean_s1'])
    X_train['clean_s2'] = str(X_train['clean_s2'])

    X_val['clean_s1'] = str(X_val['clean_s1'])
    X_val['clean_s2'] = str(X_val['clean_s2'])

    X_test['clean_s1'] = str(X_test['clean_s1'])
    X_test['clean_s2'] = str(X_test['clean_s2'])

    train_s1_seq = t.texts_to_sequences(X_train['clean_s1'].values)
    train_s2_seq = t.texts_to_sequences(X_train['clean_s2'].values)
    val_s1_seq = t.texts_to_sequences(X_val['clean_s1'].values)
    val_s2_seq = t.texts_to_sequences(X_val['clean_s2'].values)
    test_s1_seq = t.texts_to_sequences(X_test['clean_s1'].values)
    test_s2_seq = t.texts_to_sequences(X_test['clean_s2'].values)

    len_vec = [len(sent_vec) for sent_vec in train_s1_seq]

    numpy.max(len_vec)


    len_vec = [len(sent_vec) for sent_vec in train_s2_seq]
    numpy.max(len_vec)
    max_len = 50

    train_s1_seq = pad_sequences(train_s1_seq, maxlen=max_len, padding='post')

    train_s2_seq = pad_sequences(train_s2_seq, maxlen=max_len, padding='post')
    val_s1_seq = pad_sequences(val_s1_seq, maxlen=max_len, padding='post')
    val_s2_seq = pad_sequences(val_s2_seq, maxlen=max_len, padding='post')
    test_s1_seq = pad_sequences(test_s1_seq, maxlen=max_len, padding='post')
    test_s2_seq = pad_sequences(test_s2_seq, maxlen=max_len, padding='post')

    not_present_list = []
    vocab_size = len(t.word_index) + 1
    print('Loaded %s word vectors.' % len(fast.get_words()))
    embedding_matrix = numpy.zeros((vocab_size, len(fast['no'])))
    for word, i in t.word_index.items():
        if word in fast:
            embedding_vector = fast[word]
        else:
            not_present_list.append(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = numpy.zeros(100)

    input_1 = Input(shape=(train_s1_seq.shape[1],))
    input_2 = Input(shape=(train_s2_seq.shape[1],))

    common_embed = Embedding(name='synopsis_embedd',input_dim =len(t.word_index)+1, 
                        output_dim=len(fast['no']),weights=[embedding_matrix], 
                        input_length=train_s1_seq.shape[1],trainable=False) 
    lstm_1 = common_embed(input_1)
    lstm_2 = common_embed(input_2)


    common_lstm = LSTM(64,return_sequences=True, activation='relu')
    vector_1 = common_lstm(lstm_1)
    vector_1 = Flatten()(vector_1)

    vector_2 = common_lstm(lstm_2)
    vector_2 = Flatten()(vector_2)

    x3 = Subtract()([vector_1, vector_2])
    x3 = Multiply()([x3, x3])

    x1_ = Multiply()([vector_1, vector_1])
    x2_ = Multiply()([vector_2, vector_2])
    x4 = Subtract()([x1_, x2_])
        
    x5 = Lambda(cosine_distance, output_shape=cos_dist_output_shape)([vector_1, vector_2])
        
    conc = Concatenate(axis=-1)([x5,x4, x3])

    x = Dense(100, activation='relu', name='conc_layer')(conc)
    x = Dropout(0.01)(x)
    out = Dense(1, activation='sigmoid', name = 'out')(x)

    model = Model([input_1, input_2], out)

    model.compile(loss='binary_crossentropy', metrics=['acc', tf.keras.metrics.AUC()], optimizer='adam')
    model.summary()
    model.fit([train_s1_seq,train_s2_seq],y_train.values.reshape(-1,1), epochs = 5,
            batch_size=64,validation_data=([val_s1_seq, val_s2_seq],y_val.values.reshape(-1,1)))
    model.save('model.bin')
    #saving
    with open('tokenizer.pickle', 'wb') as f:
        pickle.dump(t, f, protocol=pickle.HIGHEST_PROTOCOL)
    return model
def check_model():
    if os.path.exists('model.bin'):
        print('loading model...')
        model = keras.models.load_model('model.bin')
        with open('tokenizer.pickle', 'rb') as f:
            t = pickle.load(f)
    else:
        print('creating model...')
        model = create_model()

def get_vectors(text1):
    _, text1 = compute(text1)
    sequence = t.texts_to_sequences([text1])
    sequence = pad_sequences(sequence, maxlen=50, padding='post')
    return sequence


def get_similarity(text1, text2):
    in1 = get_vectors(text1)
    in2 = get_vectors(text2)
    result = numpy.array(model([in1,in2]))
    return result[0][0]

