import json
import numpy as np
import pickle
from keras.models import Sequential 
from keras.layers import Dense, Embedding, GlobalAveragePooling1D
from keras.preprocessing.text import Tokenizer
from keras import utils
from sklearn.preprocessing import LabelEncoder

with open('intent.json') as file:
    data = json.load(file)
training_sentences = []
training_labels = []
labels =[]
response = []

for intent in data['intents']:
    for pattern in intent['pattern']:
        training_sentences.append(pattern)
    # training_sentences.append(intent['pattern'])
        training_labels.append(intent['tag'])
    response.append(intent['response'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])
num_classes = len(labels)

# converting an arraylike variable to array
LblEncoder = LabelEncoder()
LblEncoder.fit(training_labels)
training_labels1 = LblEncoder.transform(training_labels)

Lbl_Encoder = LabelEncoder()
Lbl_Encoder.fit(training_labels)

vocab_size = 1000
embedding_dim = 16
max_len = 20
oov_token = "<OOV>" 

tokenizer = Tokenizer(vocab_size, oov_token= oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = utils.pad_sequences(sequences,max_len,truncating='post')

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam', metrics=['accuracy'])
# model.summary()

# training data
epochs = 500
history = model.fit(padded_sequences, np.array(training_labels1), epochs=epochs)

# model.save('chat_model')
# # to save the fitted tokenizer
# with open('tokenizer.pickle', 'wb') as handle:
#     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# # to save the fitted label encoder
# with open('label_encoder.pickle', 'wb') as ecn_file:
#     pickle.dump(LblEncoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)