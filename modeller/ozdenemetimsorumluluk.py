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
            self.y = LabelEncoder().fit_transform(self.data["label"])
            self.X = vectorizer.fit_transform(self.data["text"])
            self.model = RandomForestClassifier(bootstrap=True, criterion="gini", max_features=0.2, min_samples_leaf=4,
                                                min_samples_split=6, n_estimators=100).fit(self.X, self.y)

            self.text = vectorizer.transform([text])
            self.predict()

        def predict(self):
            """ Predicts the label of the text"""
            return self.model.predict(self.text)

    class Sorumluluk:
        """Özdenetim sorumluluk modeli sorumluluk alt ölçeği için kullanılır"""

        def __init__(self, text=None, vectorizer=TfidfVectorizer(analyzer="char", ngram_range=(2, 3))):
            """ X: text
                y: label encoded labels
                text: text to predict"""
            self.data = DataHandler(
                "data/öz denetim sorumluluk/sorumluluk vs heyecan arama/sorumluluk_vs_heyecanarama.csv").get_data()
            self.X = vectorizer.fit_transform(self.data["text"])
            self.y = LabelEncoder().fit_transform(self.data["label"])
            self.model = RandomForestClassifier(bootstrap=True, criterion="gini", max_features=0.2, min_samples_leaf=4,
                                                min_samples_split=6, n_estimators=100).fit(self.X, self.y)

            self.text = vectorizer.transform([text])
            self.predict()

        def predict(self):
            """ Predicts the label of the text"""
            return self.model.predict(self.text)

    class KurallarabaglilikVeHeyecanarama:
        """Özdenetim sorumluluk modeli kurallarabaglilik ve heyecanarama alt ölçeği için kullanılır"""
        def __init__(self, text=None, vectorizer=TfidfVectorizer(analyzer="char", ngram_range=(2, 3))):
            """ X: text
                y: label encoded labels
                text: text to predict"""
            self.data = DataHandler(
                "data/öz denetim sorumluluk/sorumluluk vs heyecan arama/sorumluluk_vs_heyecanarama.csv").get_data()
            self.y = LabelEncoder().fit_transform(self.data["label"])
            self.X = vectorizer.fit_transform(self.data["text"])
            self.model = RandomForestClassifier(bootstrap=True, criterion="gini", max_features=0.25, min_samples_leaf=3,
                                                min_samples_split=11, n_estimators=100).fit(self.X, self.y)

            self.text = vectorizer.transform([text])
            self.predict()

        def predict(self):
            """ Predicts the label of the text"""
            return self.model.predict(self.text)


