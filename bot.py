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
