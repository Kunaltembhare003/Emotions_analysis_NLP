import pandas as pd
import pathlib
import sys
from tensorflow.keras.preprocessing.text import Tokenizer
#from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

def load_and_split_data(data_path_train, data_path_test):
    train = pd.read_csv(data_path_train)
    test = pd.read_csv(data_path_test)
    X_train = train['text'].astype(str)
    y_train = train['label']
    X_test = test['text'].astype(str)
    y_test = test['label']
    return X_train, X_test, y_train, y_test

# tokenization of seq
def tokenization(X_train, X_test, y_train, y_test, output_path):
    # Tokenize the text data
    tokenizer = Tokenizer(num_words=50000)
    tokenizer.fit_on_texts(X_train)
    tokenizer.fit_on_texts(X_test)
    tokenizer_json = tokenizer.to_json()
    with open(output_path+'/tokenizer.json', 'w') as f:
        json.dump(tokenizer_json, f)

def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    input_file_train = sys.argv[1]
    input_file_test = sys.argv[2]
    data_path_train = home_dir.as_posix() + input_file_train
    data_path_test = home_dir.as_posix() + input_file_test
    output_path = home_dir.as_posix() + '/data/interim'
    
    X_train, X_test, y_train, y_test = load_and_split_data(data_path_train, data_path_test)
    tokenization(X_train, X_test, y_train, y_test, output_path )

if __name__=='__main__':
    main()

    
    

