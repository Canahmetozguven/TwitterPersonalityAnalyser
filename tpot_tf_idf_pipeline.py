import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from skompiler import skompile
from modeller.data_handler import DataHandler
import pickle
# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = DataHandler('data/yumuşak başlılık/tepki_vs_yumuşakkalplilik.csv').get_data()
X = TfidfVectorizer().fit_transform(tpot_data["text"])
y =  LabelEncoder().fit_transform(tpot_data["label"])

# Average CV score on the training set was: 0.9771521930375769
exported_pipeline =  make_pipeline(
                RFE(estimator=ExtraTreesClassifier(criterion="entropy", max_features=0.8500000000000001,
                                                   n_estimators=100), step=0.7500000000000001),
                LinearSVC(C=25.0, dual=True, loss="squared_hinge", penalty="l2", tol=0.01))

exported_pipeline.fit(X, y)

pickle.dump(exported_pipeline, open("modeller/yumuşaklık", 'wb'))