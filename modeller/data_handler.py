import pandas as pd
import nltk
from nltk.corpus import stopwords


nltk.download('stopwords')
SW = stopwords.words("turkish")


class DataHandler:
    def __init__(self, data_path, text_column="text"):
        self.data_path = data_path
        self.data = self._load_data()
        self.text_column = text_column
        self.data = self.data_wrangler()

    def _load_data(self):
        return pd.read_csv(self.data_path)

    def data_wrangler(self):
        self.data[f"{self.text_column}"] = self.data[f"{self.text_column}"].str.lower()
        self.data[f"{self.text_column}"] = self.data[f"{self.text_column}"].str.replace(r'[^\w\s]+', " ", regex=True)
        self.data[f"{self.text_column}"] = self.data[f"{self.text_column}"].str.replace(r'\d+', " ", regex=True)
        self.data[f"{self.text_column}"] = self.data[f"{self.text_column}"].apply(
            lambda words: ' '.join(word.lower() for word in str(words).split() if word not in SW))
        return self.data
    def get_data(self):
        return self.data
