from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization
import numpy as np
import pandas as pd

class ToxicCommentClassifier:
    def __init__(self, model_path='model3.h5', max_words=200000, output_sequence_length=1800, csv_path='../jigsaw-toxic-comment-classification-challenge/train.csv/train.csv'):
        self.model_path = model_path
        self.max_words = max_words
        self.output_sequence_length = output_sequence_length
        self.csv_path = csv_path
        self.model = self.load_model()
        self.vectorizer = self.create_vectorizer()
        

    def load_model(self):
        return load_model(self.model_path)

    def create_vectorizer(self):
        df = pd.read_csv(self.csv_path)
        x = df['comment_text']
        vectorizer = TextVectorization(
            max_tokens=self.max_words,
            output_sequence_length=self.output_sequence_length,
            output_mode='int'
        )
        vectorizer.adapt(x.values)
        return vectorizer

    def predict_category(self, input_text):
        input_text_vectorized = self.vectorizer(input_text)
        output_arr = self.model.predict(np.expand_dims(input_text_vectorized, 0))[0]
        return output_arr

    def classify_comment(self, input_text):
        output_arr = self.predict_category(input_text)
        categories = ["toxic", "severely toxic", "obscene", "threatful", "insultful", "identity hateful"]
        output = ""


        for i, category in enumerate(categories):
            if 0.8 <= output_arr[i] < 1:
                output+=f"Highly {category}, "
            elif 0.5 <= output_arr[i] < 0.8:
                output+=f"Potentially {category}, "

        if output=="":
            return "Ok"
        return output


