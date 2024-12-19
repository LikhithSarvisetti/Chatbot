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