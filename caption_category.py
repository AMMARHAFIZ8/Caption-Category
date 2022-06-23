# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 14:44:22 2022

@author: ACER
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 14:03:48 2022

@author: ACER
"""
import numpy as np
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import pickle
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# 1) trained model --> loading from h5
# 2) tokenizer
# 3) MMS?OHE --> loading pickle

#%% Ammar
# Deployment ususally done on another PC/mobile

# to load trained model 
loaded_model = load_model(os.path.join(os.getcwd(),"model.h5"))

loaded_model.summary()

#to load tokenozer
TOKENIZER_PATH = os.path.join(os.getcwd(), 'tokenizer_category.json')
with open (TOKENIZER_PATH,'r') as json_file:
    loaded_tokenizer = json.load(json_file)


OHE_PATH = os.path.join(os.getcwd(), 'ohe.pkl')
# #to load ohe
with open(OHE_PATH,'rb')as file:
    loaded_ohe = pickle.load(file)



# Preprocessing

while True: # Forever loop #beware may overload
    input_text = input(" text : ")
    
    input_text = re.sub('<.*?>',' ',input_text)
    input_text = re.sub('[^a-zA-Z]',' ',input_text).lower().split()
    
    #ammar
    tokenizer = tokenizer_from_json(loaded_tokenizer)
    input_text_encoded = tokenizer.texts_to_sequences(input_text)
    
    input_text_encoded = pad_sequences(np.array(input_text_encoded).T,
                                         maxlen=180,
                                         padding='post',truncating='post')
    
    outcome = loaded_model.predict(np.expand_dims(input_text_encoded,axis=-1))
    
    
    print("Ammar's model says the text is " + loaded_ohe.inverse_transform(outcome))
    
