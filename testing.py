import json
import numpy as np 
import pickle
import colorama
colorama.init()
from colorama import Fore, Style
from tensorflow import keras
from chatbot import Lbl_Encoder, tokenizer, model  

with open('intent.json') as file:
    data = json.load(file)

def chat():
    # # load trained model
    # modellk = keras.models.load_model('chat_model')

    # # load tokenizer object
    # with open('tokenizer.pickle', 'rb') as handle:
    #     tokeniizer = pickle.load(handle)

    # # load label encoder object
    # with open('label_encoder.pickle', 'rb') as enc:
    #     lbl_encoder = pickle.load(enc)

    # parameters
    max_len = 20
    
    while True:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == "quit":
            break

        result = model.predict(keras.utils.pad_sequences(tokenizer.texts_to_sequences([inp]),truncating='post', maxlen=max_len))
        # print(result)
      
        tag = Lbl_Encoder.inverse_transform([np.argmax(result)])
        # print(tag)
        for i in data['intents']:
            # print(i["tag"])

            if i["tag"] == tag:
                print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL , np.random.choice(i['response']))

print(Fore.YELLOW + "Start messaging with the bot (type quit to stop)!" + Style.RESET_ALL)
chat()
