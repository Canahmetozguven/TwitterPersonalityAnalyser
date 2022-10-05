import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from .data_handler import DataHandler


class OzdenetimSorumlulukModel:
    class Duzenlilik:

        """Özdenetim sorumluluk modeli düzenlilik alt ölçeği için kullanılır"""

        def __init__(self, text=None, vectorizer=TfidfVectorizer(analyzer="char", ngram_range=(2, 3))):
            """ X: text
                y: label encoded labels
                text: text to predict"""
            self.data = DataHandler(
                "data/öz denetim sorumluluk/düzenlilik vs heyecan arama/düzenlilik_vs_heyecanarama.csv").get_data()
            self.X = vectorizer.fit_transform(self.data["text"])
            self.text = vectorizer.transform([text])
            self.predict()

        def predict(self):
            """ Predicts the label of the text"""
            x = self.text
            return pickle.load(open("modeller/düzenlilik", 'rb')).predict(x)

    class Sorumluluk:
        """Özdenetim sorumluluk modeli sorumluluk alt ölçeği için kullanılır"""

        def __init__(self, text=None, vectorizer=TfidfVectorizer(analyzer="char", ngram_range=(2, 3))):
            """ X: text
                y: label encoded labels
                text: text to predict"""
            self.data = DataHandler(
                "data/öz denetim sorumluluk/sorumluluk vs heyecan arama/sorumluluk_vs_heyecanarama.csv").get_data()
            self.X = vectorizer.fit_transform(self.data["text"])
            self.text = vectorizer.transform([text])
            self.predict()

        def predict(self):
            """ Predicts the label of the text"""
            x = self.text
            return pickle.load(open("modeller/sorumluluk", 'rb')).predict(x)
    class KurallarabaglilikVeHeyecanarama:
        """Özdenetim sorumluluk modeli kurallarabaglilik ve heyecanarama alt ölçeği için kullanılır"""
        def __init__(self, text=None, vectorizer=TfidfVectorizer(analyzer="char", ngram_range=(2, 3))):
            """ X: text
                y: label encoded labels
                text: text to predict"""
            self.data = DataHandler(
                "data/öz denetim sorumluluk/kurallara bağlılık vs heyecan arama/kurallarabağlılık_vs_heyecanarama.csv").get_data()
            self.X = vectorizer.fit_transform(self.data["text"])
            self.text = vectorizer.transform([text])
            self.predict()

        def predict(self):
            """ Predicts the label of the text"""
            x = self.text
            return pickle.load(open("modeller/kurallarabağlılık", 'rb')).predict(x)



