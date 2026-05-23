import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer 

lemmatizer = WordNetLemmatizer()

intent = json.loads(open(r'D:\project\NLP\DATA\intents.json').read())

words = []
classes = []    
documents = []
ignore_letters = ['?', '!', '.', ',']   

for intent in intent['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag']) 

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))  
classes = sorted(set(classes))  
pickle.dump(words, open(r'D:\project\NLP\DATA\words.pkl', 'wb'))
pickle.dump(classes, open(r'D:\project\NLP\DATA\classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)
train_x = list(training[:, 0])
train_y = list(training[:, 1])
