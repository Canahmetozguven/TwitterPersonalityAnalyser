from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from data_handler import DataHandler


class YumukBaslilik:
    class SakinlikVeTepkisellik:
        data = DataHandler("data/yumuşak başlılık/sakinlik_vs_tepkisellik.csv").get_data()
        y = LabelEncoder().fit_transform(data["label"])
        X = data["text"]
        """Yumuk baslilik modeli sakinlik ve tepkisellik alt ölçeği için kullanılır"""

        def __init__(self, X=X, y=y, text=None, vectorizer=TfidfVectorizer(analyzer="char", ngram_range=(2, 3))):
            """ X: text
                y: label encoded labels
                text: text to predict"""
            self.X = vectorizer.fit_transform(X)
            self.model = RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.2,
                                                min_samples_leaf=6, min_samples_split=6, n_estimators=100).fit(self.X,
                                                                                                               y)

            self.text = vectorizer.transform(text)
            self.predict()

        def predict(self):
            """ Predicts the label of the text"""
            return self.model.predict(self.text)

    class Yumusaklik:
        data = DataHandler("data/yumuşak başlılık/yumuşaklık.csv").get_data()
        y = LabelEncoder().fit_transform(data["label"])
        X = data["text"]
        """Yumuk baslilik modeli yumusaklik alt ölçeği için kullanılır"""

        def __init__(self, X=X, y=y, text=None, vectorizer=TfidfVectorizer()):
            """ X: text
                y: label encoded labels
                text: text to predict"""
            self.X = vectorizer.fit_transform(X)
            self.model = make_pipeline(
                RFE(estimator=ExtraTreesClassifier(criterion="entropy", max_features=0.8500000000000001,
                                                   n_estimators=100), step=0.7500000000000001),
                LinearSVC(C=25.0, dual=True, loss="squared_hinge", penalty="l2", tol=0.01)).fit(self.X, y)

            self.text = vectorizer.transform(text)
            self.predict()

        def predict(self):
            """ Predicts the label of the text"""
            return self.model.predict(self.text)
