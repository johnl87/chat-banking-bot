import random
import json
import pickle
import numpy as np
from tensorflow import keras



import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer

from keras.models import load_model
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

#read in binary mode - rb
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

model = load_model('chatbot_model.h5')

def tidy_up_sentence(string):
    sentence_words = nltk.word_tokenize(string)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def outcome_of_words(string):
    sentence_words = tidy_up_sentence(string)
    bag = len(words) * [0] 
    for x in sentence_words:
        for y, word in enumerate(words):
            if word == x:
                bag[y] = 1
            else:
                bag[y] = 0
    
    return np.array(bag)

def predict_intent(string):
    #cow = collection of words
   cow = outcome_of_words(string)
   result = model.predict(np.array([cow]))[0]
   ERROR_THRESHOLD = 0.25
   results = [[i , r] for i, r in enumerate(result) if r > ERROR_THRESHOLD]

   results.sort(key = lambda x: x[1], reverse=True)
   return_list = []
   for r in results:
    return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_user_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for x in list_of_intents:
        if x['tag'] == tag:
            result = random.choice(x['responses'])
            break
    return result

print("bot is live, all clear to answer your questions!")

while True:
    result = get_user_response(predict_intent(input("")), intents)
    print(result)
