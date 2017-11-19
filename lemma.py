__author__ = 'vijetasah'

import nltk
import pickle

from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer()

def words(fname):
    lemmatized_data = ""
    original_data = ""
    with open(fname, 'r') as document:
        for line in document:
            for word in line.strip().split():
                original_data = original_data + " "+ word
                lemmatized_data =lemmatized_data + " " +lemma.lemmatize(word, pos="v")
    #print("original text : ",original_data)
    print(lemmatized_data)


with open('BookEcclesiastes.pickle', 'rb') as f:
        x = pickle.load(f)

words(x)