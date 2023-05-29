import json
import numpy as np 
import random
import pickle
import colorama
colorama.init()
from colorama import Fore, Style, Back
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from chatbot import Lbl_Encoder, tokenizer, model, training_labels 

with open('intent.json') as file:
    data = json.load(file)

def chat():
    # load trained model
    modellk = keras.models.load_model('chat_model')

    # load tokenizer object
    with open('tokenizer.pickle', 'rb') as handle:
        tokeniizer = pickle.load(handle)

    # load label encoder object
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    # parameters
    max_len = 20
    
    while True:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == "quit":
            break
        
        # tokenizer = Tokenizer(1000, oov_token= "<OOV>")
        # tokenizer.fit_on_texts([inp])

        # model = Sequential()

        result = model.predict(keras.utils.pad_sequences(tokenizer.texts_to_sequences([inp]),truncating='post', maxlen=max_len))
        # print(result)
        # lbl_encoder = LabelEncoder()
        # lbl_encoder.fit([np.argmax(result)])
        
        tag = Lbl_Encoder.inverse_transform([np.argmax(result)])
        # print(tag)
        for i in data['intents']:
            # print(i["tag"])

            if i["tag"] == tag:
                print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL , np.random.choice(i['response']))

#         # print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL,random.choice(responses))

print(Fore.YELLOW + "Start messaging with the bot (type quit to stop)!" + Style.RESET_ALL)
chat()