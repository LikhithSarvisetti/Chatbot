import json
import string
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer #It has ability to lemmatize
import tensorflow as tf
from tensorflow.keras import Sequential #sequential groups a linear stack of layers
from tensorflow.keras.layers import Dense, Dropout
nltk.download("punkt")
nltk.download("wordnet")
nltk.download('omw-1.4')
nltk.download('punkt_tab')

with open('data.json', 'r') as file:
    data = json.load(file)

lemmatizer = WordNetLemmatizer()
words, classes, doc_X, doc_y = [], [], [], []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        doc_X.append(pattern)
        doc_y.append(intent["tag"])
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
words, classes = sorted(set(words)), sorted(set(classes))

training = []
out_empty = [0] * len(classes)
for idx, doc in enumerate(doc_X):
    bow = [1 if word in nltk.word_tokenize(doc.lower()) else 0 for word in words]
    output_row = list(out_empty)
    output_row[classes.index(doc_y[idx])] = 1
    training.append([bow, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)
train_X = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1])) 

input_shape, output_shape = (len(train_X[0]),), len(train_y[0])
model = Sequential()
model.add(Dense(128, input_shape=input_shape, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(output_shape, activation="softmax"))

adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])
print(model.summary())
model.fit(x=train_X, y=train_y, epochs=200, verbose=1)

def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def bag_of_words(text, vocab):
    tokens = clean_text(text)
    bow = [1 if word in tokens else 0 for word in vocab]
    return np.array(bow)

def pred_class(text, vocab, labels):
    bow = bag_of_words(text, vocab)
    result = model.predict(np.array([bow]))[0]
    thresh = 0.2
    y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]
    y_pred.sort(key=lambda x: x[1], reverse=True)
    return [labels[r[0]] for r in y_pred]

def get_response(intents_list, intents_json):
    tag = intents_list[0]
    for i in intents_json["intents"]:
        if i["tag"] == tag:
            return random.choice(i["responses"]) 

while True:
    message = input("")
    intents = pred_class(message, words, classes)
    result = get_response(intents, data)
    print(result)
    break