import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from .data_handler import DataHandler


# Average CV score on the training set was: 0.97
class DisadonuklulukModel:
    class CanlilikModel:
        """Dışa dönüklük modeli canlılık alt ölçeği için kullanılır"""

        def __init__(self, text=None, vectorizer=TfidfVectorizer()):
            """ X: text
                y: label encoded labels
                text: text to predict"""
            self.data = DataHandler("data/dışa dönüklük/canlılık/canlılık_vs_içedönüklük.csv").get_data()
            self.y = LabelEncoder().fit_transform(self.data["label"])
            self.X = vectorizer.fit_transform(self.data["text"])
            self.model = RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.2,
                                                min_samples_leaf=6,
                                                min_samples_split=6, n_estimators=100).fit(self.X, self.y)
            self.text = vectorizer.transform([text])
            self.predict()

        def predict(self):
            """ Predicts the label of the text"""
            return self.model.predict(self.text)

    class IcedonuklukModelveGiriskenlikModel:

        """Dışa dönüklük modeli içedönüklük ve girişkenlik alt ölçeği için kullanılır"""

        def __init__(self, text=None, vectorizer=CountVectorizer()):
            """ X: text
                y: label encoded labels
                text: text to predict"""
            self.data = DataHandler("data/dışa dönüklük/girişkenlik vs içe dönüklük/girişkenlik_vs_içedönüklük.csv").get_data()
            self.X = vectorizer.fit_transform(self.data["text"])
            self.y = LabelEncoder().fit_transform(self.data["label"])
            self.model = LinearSVC(C=0.1, dual=True, loss="squared_hinge", penalty="l2", tol=0.0001).fit(self.X, self.y)
            self.text = vectorizer.transform([text])
            self.predict()

        def predict(self):
            """ Predicts the label of the text"""
            return self.model.predict(self.text)
