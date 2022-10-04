from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

from .data_handler import DataHandler

class DuygusalDengeModel:
    class EndiseyeYatkinlikVeKendineGuven:

        """Duygusal denge modeli endişeye yatkınlık ve kendine güven alt ölçeği için kullanılır"""

        def __init__(self, text=None, vectorizer=CountVectorizer()):
            """ X: text
                y: label encoded labels
                text: text to predict"""
            self.data = DataHandler("data/duygusal denge/endişeyeyatkınlık_vs_kendine+güven.csv").get_data()
            self.X = vectorizer.fit_transform(self.data.text)
            self.y = LabelEncoder().fit_transform(self.data["label"])
            self.model = LogisticRegression(C=20.0, dual=False, penalty="l2").fit(self.X, self.y)
            self.text = vectorizer.transform([text])
            self.predict()

        def predict(self):
            """ Predicts the label of the text"""
            return self.model.predict(self.text)

