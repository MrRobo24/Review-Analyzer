# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 21:43:51 2020

@author: mr_ro
"""

import numpy as np
import pandas as pd
from scipy import sparse
import pickle
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import roc_auc_score

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, SpatialDropout1D, GRU
from keras.layers import Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

#loading data
print("LOG: LOADING DATA")
#business = pd.read_csv("C:\\Users\\mr_ro\\Desktop\\Review Analyzer\\model\\yelp_business.csv")
#review_all = pd.read_csv("C:\\Users\\mr_ro\\Desktop\\Revpiew Analyzer\\model\\yelp_review.csv")

'''
The following files are around 3 GB download them from kaggle or official portals and extract
the CSVs at appropriate location.
'''

business = pd.read_csv("../data/yelp_business.csv")
review_all = pd.read_csv("../data/yelp_review.csv")
print("LOG: DATA LOADED")

a = business[business['categories'].str.contains('Restaurant') == True]
rev = review_all[review_all.business_id.isin(a['business_id']) == True]

rev_samp = rev.sample(n = 350000, random_state = 42)
train = rev_samp[0:280000]
test = rev_samp[280000:]

train = train[['text', 'stars']]
train['stars'].hist();train.head()

train = pd.get_dummies(train, columns = ['stars'])
train.head()

test = test[['text', 'stars']]
test = pd.get_dummies(test, columns = ['stars'])
train.shape, test.shape

#taking small sample of train and test set to fine tune the model
#35% of actual train and test set
train_samp = train.sample(frac = .35, random_state = 40)
test_samp = test.sample(frac = .35, random_state = 40)
print(train_samp.shape, test_samp.shape)
'''
if the yelp_review dataset and yelp_business file is missing then you can load the above 
sample files directly from data folder.
The train sample file has 98000 reviews inside it and test sample file has 24500 reviews in it.
'''

#NN model
# In this model I am using Glove pretrained word vectors for word embeddings
embed_size = 200 #according to the glove file we are using 
# max number of unique words 
max_features = 20000
# max number of words from review to use
maxlen = 200

'''
the file is too big too keep along with the project
hence you can download the glove.twitter.27B.200d.txt file
and paste it at appropriate location
'''
#embedding_file = "C:\\Users\\mr_ro\\Desktop\\Review Analyzer\\model\\glove.twitter.27B.200d.txt"
embedding_file = "../data/glove.twitter.27B.200d.txt"


# read in embeddings
def get_coefs(word,*arr): 
    return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(embedding_file, encoding="utf8"))

class_names = ['stars_1', 'stars_2', 'stars_3', 'stars_4', 'stars_5']
# Splitting off my y variable
y = train_samp[class_names].values

#tokenizing and padding the sequences

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_samp['text'].values))

# saving tokenizer
with open('tokenizer_2.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

X_train = tokenizer.texts_to_sequences(train_samp['text'].values)
X_test = tokenizer.texts_to_sequences(test_samp['text'].values)
x_train = pad_sequences(X_train, maxlen = maxlen)
x_test = pad_sequences(X_test, maxlen = maxlen)

print("LOG: SEQUENCES EMBEDDED AND PADDED")

#working for the words which don't exist in the Glove data
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
# create a zeros matrix of the correct dimensions 
embedding_matrix = np.zeros((nb_words, embed_size))
missed = []
for word, i in word_index.items():
    if i >= max_features: break
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        missed.append(word)

#experimentation
len(missed)

missed[0:10]

missed[1000:1010]


#defining the model
inp = Input(shape = (maxlen,))
x = Embedding(max_features, embed_size, weights = [embedding_matrix], trainable = True)(inp)
x = SpatialDropout1D(0.5)(x)
x = Bidirectional(LSTM(40, return_sequences=True))(x)
x = Bidirectional(GRU(40, return_sequences=True))(x)

avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
conc = concatenate([avg_pool, max_pool])
outp = Dense(5, activation = 'sigmoid')(conc)

model = Model(inputs = inp, outputs = outp)
earlystop = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 3)
checkpoint = ModelCheckpoint(monitor = 'val_loss', save_best_only = True, filepath = 'yelp_lstm_gru_weights.hdf5')
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

#fitting the model
model.fit(x_train, 
          y, 
          batch_size = 512, 
          epochs = 20, 
          validation_split = .1,
          callbacks=[earlystop, checkpoint])

# evaluating the model
y_test = model.predict([x_test], batch_size=1024, verbose = 1)

model.evaluate(x_test, test_samp[class_names].values, verbose = 1, batch_size=1024)
'''
v = metrics.classification_report(np.argmax(test_samp[class_names].values, axis = 1),np.argmax(y_test, axis = 1))
print(v)
'''

model.save("model2_2.h5")


train_samp.to_csv("../data/train_samp_35.csv")
test_samp.to_csv("../data/test_samp_35.csv")
