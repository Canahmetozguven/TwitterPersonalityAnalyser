import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from .data_handler import DataHandler
import numpy as np


class YumukBaslilik:
    class SakinlikVeTepkisellik:


        """Yumuk baslilik modeli sakinlik ve tepkisellik alt ölçeği için kullanılır"""

        def __init__(self, text=None, vectorizer=TfidfVectorizer(analyzer="char", ngram_range=(2, 3))):
            """ X: text
                y: label encoded labels
                text: text to predict"""
            self.data = DataHandler("data/yumuşak başlılık/sakinlik_vs_tepkisellik.csv").get_data()
            self.X = self.data["text"]
            self.X = vectorizer.fit_transform(self.X)
            self.text = vectorizer.transform([text])
            self.predict()

        def predict(self):
            """ Predicts the label of the text"""
            x = self.text
            return pickle.load(open("modeller/sakinlikvetepkisellik", 'rb')).predict(x)


    class Yumusaklik:


        """Yumuk baslilik modeli yumusaklik alt ölçeği için kullanılır"""

        def __init__(self, text=None, vectorizer=TfidfVectorizer()):
            """ X: text
                y: label encoded labels
                text: text to predict"""
            self.data = DataHandler("data/yumuşak başlılık/tepki_vs_yumuşakkalplilik.csv").get_data()
            self.X = vectorizer.fit_transform(self.data["text"])
            self.model = pickle.load(open("modeller/yumuşaklık", 'rb'))

            self.text = vectorizer.transform([text])
            self.predict()

        def predict(self):
            """ Predicts the label of the text"""
            return self.model.predict(self.text)
