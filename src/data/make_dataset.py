import pandas as pd
import pathlib
import sys
import yaml
from pathlib import Path
# from textacy import preprocessing
from nltk.stem.snowball import SnowballStemmer
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re

# Download nltk resources
nltk.download('punkt')
nltk.download('stopwords') 
nltk.download('wordnet')

# Load datasets
def load_data(input_file):
    df =  pd.read_csv(input_file)
    return df
# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()
# Define a function to lemmatize a single word
def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def preprocessing_data(df):
    # Step I: Remove URL's
    df["text"] = df["text"].str.replace(r'https\S+', '', regex=True)
    # Step II: Remove special character and punctuation
    df["text"]= df['text'].str.replace(r'[^\w\s]','', regex=True)
    # Step III: Remove extra whitespace
    df["text"]= df["text"].str.replace(r'\s+',' ',regex=True )
    # Step IV: Remove Numeric values in text
    df['text'] = df['text'].str.replace(r'\d+', '', regex=True)
    # Step V: Lower the text cases
    df['text']= df['text'].str.lower()
    # Step VI: Remove stopwords
    stop = stopwords.words('english')
    df['text'] = df['text'].apply(lambda x: ' '.join([word for  word in x.split() if word not in (stop)]))
    # Step VII: Remove Non-alpha mumeric
    df['text']= df["text"].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
    # Step VIII: Lemmatization

    # Apply the lemmatization function to the 'text' column
    df['text'] = df['text'].apply(lambda x: lemmatize_text(x))
    return df

def split_data(df, test_split, seed):
    # Split the dataset into train and test sets
    train, test = train_test_split(df, test_size=test_split, random_state=seed)
    return train, test


def save_data(train, test, output_path):
    # Save the split datasets to the specified output path
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    train.to_csv(output_path + '/train.csv', index=False)
    test.to_csv(output_path + '/test.csv', index=False)


def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    print(home_dir)
    params_file = home_dir.as_posix() + '/params.yaml'
    params = yaml.safe_load(open(params_file))["make_data"]

    input_file = sys.argv[1]
    print(input_file)
    data_path = home_dir.as_posix() + input_file
    print(data_path)
    output_path = home_dir.as_posix() + '/data/processed'
    print(output_path)
    
    data = load_data(data_path)
    pro_data = preprocessing_data(data)
    train_data, test_data = split_data(pro_data, params['test_split'], params['seed'])
    save_data(train_data, test_data , output_path)


if __name__ == '__main__':
    main()