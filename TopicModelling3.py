

from __future__ import print_function
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
from constants import BookWisdom_file, BookWisdom_studyline
import re

def fetch_data():
        """
        :return: returns a dict by chapters implemented by each book
        """
        req = open(BookWisdom_file)
        document = []
        lines = [line for line in req.readlines()]
        ch = 0
        for i,j in BookWisdom_studyline:
            text=""
            for line in lines[i:j]:
                    text+=line
            ch+=1
            document.append(text)
        return document

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic %d:" % (topic_idx))
        print (" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

documents = fetch_data()

no_features = 1000

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 20


# Run LDA
lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

no_top_words = 10

display_topics(lda, tf_feature_names, no_top_words)

import  pdb; pdb.set_trace()