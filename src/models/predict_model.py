import pandas as pd
# text preprocessing modules
from string import punctuation 
import numpy as np
# text preprocessing modules
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  # regular expression
import pathlib
import sys
import json
from os.path import dirname, join, realpath
from fastapi import FastAPI, Request
from tensorflow.keras.models import model_from_json
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
# Add necessary imports
from fastapi.responses import HTMLResponse
from fastapi import Form
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow 
from sklearn.metrics import confusion_matrix
# Download nltk resources
nltk.download('punkt')
nltk.download('stopwords') 
nltk.download('wordnet')


def load_and_split_data(data_path_test):
    test = pd.read_csv(data_path_test)
    X_test = test['text'].astype(str)
    y_test = test['label']
    return X_test, y_test
# load tokenizer
def load_tokenizer(home_path):
    with open(home_path, 'r') as f:
        tokenizer_json = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_json)
    return tokenizer

def padding_data( tokenizer, X_test):
    X_test_sequences = tokenizer.texts_to_sequences(X_test)
    # Max Len in X_train_sequences
    maxlen = 79
    # Perform padding on X_train and X_test sequences
    X_test_padded = pad_sequences(X_test_sequences, maxlen=maxlen, padding='post')
    return X_test_padded

def evaluate_model_test(model, X_test_padded, y_test):
    test_acc = model.evaluate(X_test_padded, y_test)
    # Predictions On Test For Confustion Matrix 
    y_pred = model.predict(X_test_padded)
    y_pred = np.argmax(y_pred, axis=1)
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # Log confusion matrix as an artifact using MLflow
    with mlflow.start_run():
        mlflow.log_param("Test_accuracy", test_acc[1])
        # Log confusion matrix as an image artifact
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")




def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    input_file_test = sys.argv[1]
    # load test data
    data_path_test = home_dir.as_posix() + input_file_test 
    X_test, y_test = load_and_split_data(data_path_test)
    tokenizer = load_tokenizer(home_dir.as_posix()+"/data/interim/tokenizer.json")
    X_test_padded = padding_data( tokenizer, X_test)
    # load model
    model_path = home_dir.as_posix()+'/models/model.h5'
    model = load_model(model_path)
    evaluate_model_test(model, X_test_padded, y_test)


if __name__ == "__main__":
    main()
