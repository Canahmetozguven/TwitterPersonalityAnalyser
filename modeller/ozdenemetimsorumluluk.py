import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from data_handler import DataHandler


class OzdenetimSorumluluk:
    class Duzenlilik:
        data = DataHandler("data/öz denetim sorumluluk/duzenlilik/duzenlilik.csv").get_data()
        y = LabelEncoder().fit_transform(data["label"])
        X = data["text"]
        """Özdenetim sorumluluk modeli düzenlilik alt ölçeği için kullanılır"""

        def __init__(self, X=X, y=y, text=None, vectorizer=TfidfVectorizer(analyzer="char", ngram_range=(2, 3))):
            """ X: text
                y: label encoded labels
                text: text to predict"""
            self.X = vectorizer.fit_transform(X)
            self.model = RandomForestClassifier(bootstrap=True, criterion="gini", max_features=0.2, min_samples_leaf=4,
                                                min_samples_split=6, n_estimators=100).fit(self.X, y)

            self.text = vectorizer.transform(text)
            self.predict()

        def predict(self):
            """ Predicts the label of the text"""
            return self.model.predict(self.text)

    class Sorumluluk:
        data = DataHandler("data/öz denetim sorumluluk/sorumluluk vs heyecan arama/sorumluluk_vs_heyecanarama.csv").get_data()
        y = LabelEncoder().fit_transform(data["label"])
        X = data["text"]
        """Özdenetim sorumluluk modeli sorumluluk alt ölçeği için kullanılır"""

        def __init__(self, X=X, y=y, text=None, vectorizer=TfidfVectorizer(analyzer="char", ngram_range=(2, 3))):
            """ X: text
                y: label encoded labels
                text: text to predict"""
            self.X = vectorizer.fit_transform(X)
            self.model = RandomForestClassifier(bootstrap=True, criterion="gini", max_features=0.2, min_samples_leaf=4,
                                                min_samples_split=6, n_estimators=100).fit(self.X, y)

            self.text = vectorizer.transform(text)
            self.predict()

        def predict(self):
            """ Predicts the label of the text"""
            return self.model.predict(self.text)

    class KurallarabaglilikVeHeyecanarama:
        data = DataHandler("data/öz denetim sorumluluk/sorumluluk vs heyecan arama/sorumluluk_vs_heyecanarama.csv").get_data()
        y = LabelEncoder().fit_transform(data["label"])
        X = data["text"]
        """Özdenetim sorumluluk modeli kurallarabaglilik ve heyecanarama alt ölçeği için kullanılır"""
        def __init__(self, X=X, y=y, text=None, vectorizer=TfidfVectorizer(analyzer="char", ngram_range=(2, 3))):
            """ X: text
                y: label encoded labels
                text: text to predict"""
            self.X = vectorizer.fit_transform(X)
            self.model = RandomForestClassifier(bootstrap=True, criterion="gini", max_features=0.25, min_samples_leaf=3,
                                                min_samples_split=11, n_estimators=100).fit(self.X, y)

            self.text = vectorizer.transform(text)
            self.predict()

        def predict(self):
            """ Predicts the label of the text"""
            return self.model.predict(self.text)
