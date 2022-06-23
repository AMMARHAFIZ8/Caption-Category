 # -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 9:38:54 2022

@author: ACER
"""

#%%
PATH = "https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv"

#%%
import pandas as pd
import numpy as np
import re
import pickle
import datetime
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras import Input
import os
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras.layers import Bidirectional, Embedding

#%%
# EDA
# Step 1 DATA LOADING

df = pd.read_csv(PATH)
df_copy = df.copy()


# Step 2 Data Inspection
df.head(10)
df.info()
df.describe()

df['category'].unique() # to get unique targets
df['text'][5]
df['category'][5]

df.duplicated().sum() #418
df[df.duplicated()]

# <br /> tags have to be removed
# Numbers/duplicates can be filtered


# Step 3 Data Cleaning

df = df.drop_duplicates() # drop duplicate

# remove html tags
# '<br /> dj9ejdwujdpi2wjdpwp <br />'.replace('<br />',' ')
# df['text'][1].replace('<br />',' ')
text = df['text'].values #features = x
category = df['category'].values #category = y


for index,tex in enumerate(text):
    #remove html tags
    # remove ?  dont be greedy
    # zero or more occurance
    # any character except new line (/n)
    text[index] = re.sub('<.*?>',' ',tex)
    
    #convert lower case
    # remove numbers
    #^ means NOT
    text[index] = re.sub('[^a-zA-Z]',' ',tex).lower().split()


# Step 4 Features Selection
# Nothing to select


# Step 5 Preprocessing:
#       1 Convert into lower case
#       2 Tokenization
vocab_size = 10000
oov_token = 'OOV'

# Tokenization
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_token)


tokenizer.fit_on_texts(text) # to learn all of the words
word_index = tokenizer.word_index
print(word_index)

# to convert into numbers
train_sequences = tokenizer.texts_to_sequences(text) # converts into numbers

#       3 Padding & truncating

length_of_texts = [len(i)for i in train_sequences] #list comprehension
np.median(length_of_texts) # get max length  for padding
print(np.median(length_of_texts))

max_len = 100


padded_text = pad_sequences(train_sequences, maxlen=max_len, padding='post',
                              truncating='post')


#       4 One hot encoding
#only for category

ohe = OneHotEncoder(sparse=False)
category = ohe.fit_transform(np.expand_dims(category, axis=-1))

#       5 Train test split

X_train, X_test, y_train, y_test = train_test_split(padded_text, category,
                                                    test_size=0.3,
                                                    random_state=123)

X_train = np.expand_dims(X_train,axis=-1) # 2 to 3 dimensions
X_test = np.expand_dims(X_test,axis=-1)


#%% Model development

#LSTM

embedding_dim = 64


model = Sequential()
model.add(Input(100)) #np.shape(X_train)[1:]
model.add(Embedding(vocab_size, embedding_dim))
model.add(Bidirectional(LSTM(embedding_dim,return_sequences=(True))))
# model.add(LSTM(128,return_sequences=(True)))
model.add(Dropout(0.3))
model.add(LSTM(128))
model.add(Dropout(0.3))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(5,'softmax'))
model.summary()


plot_model(model,show_layer_names=(True),show_shapes=(True)) # plot flowchart of model layer

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='acc')

# Stopping Callbacks 
# Tensorboard

early_stopping_callback = EarlyStopping(monitor='loss', patience=3)

LOG_PATH = os.path.join(os.getcwd(),'Logs')

log_dir = datetime.datetime.now()
tensorboard_callback = TensorBoard(log_dir=LOG_PATH)

hist = model.fit(X_train, y_train, batch_size=128, epochs=100,
                 validation_data=(X_test, y_test),
                 callbacks=[tensorboard_callback,early_stopping_callback])

#Plot Visualisation
#loss and accuracy for each training and validation
hist.history.keys()

plt.figure()
plt.plot(hist.history['loss'],'r--', label= 'Training loss')
plt.plot(hist.history['val_loss'],'r--', label= 'Validation loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(hist.history['acc'],'r--', label = 'Training acc')
plt.plot(hist.history['val_acc'], label = 'Validation acc')
plt.legend()
plt.show()

#%% Model Evaluation
y_true = y_test
y_pred = model.predict(X_test)


y_true = np.argmax(y_true,axis=1) # to convert 0/1
y_pred = np.argmax(y_pred,axis=1) # to convert 0/1


# show accuracy_score, accuracy_score, classification_report

print(classification_report(y_true,y_pred))
print(accuracy_score(y_true, y_pred))
print(confusion_matrix(y_true,y_pred))

#%% model saving

MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.h5') #machine learning , deep learning use h5, minmax use pickle
model.save(MODEL_SAVE_PATH)

import json
token_json = tokenizer.to_json()

TOKENIZER_PATH = os.path.join(os.getcwd(),'tokenizer_category.json')
with open(TOKENIZER_PATH,'w') as file:
    json.dump(token_json,file)

OHE_PATH = os.path.join(os.getcwd(),'ohe.pkl')
with open(OHE_PATH,'wb') as file:
    pickle.dump(ohe, file)



#%% Discussion/Reporting


# -The accuracy score and f1 of this model is 88.56% score
# -Model is consider great and its learning from the training.  
# -Training graph shows an overfitting since the training accuracy is higher  than validation accuracy
     
# -This model seems not give any effect although Earlystopping with LSTM can overcome overfitting.
# -With suggestion to overcome overfitting can try other architecture like BERT, transformer or GPT3 model.
 