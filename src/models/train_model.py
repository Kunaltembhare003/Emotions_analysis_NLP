import pandas as pd
import pathlib
import sys
import joblib
import yaml
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
from keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from keras.layers import GRU, Dense, Embedding, Flatten, Dropout
from keras.activations import softmax

def load_tokenizer(home_path):
    with open(home_path, 'r') as f:
        tokenizer_json = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_json)
    return tokenizer

def load_and_split_data(data_path_train, data_path_test):
    train = pd.read_csv(data_path_train)
    test = pd.read_csv(data_path_test)
    X_train = train['text'].astype(str)
    y_train = train['label']
    X_test = test['text'].astype(str)
    y_test = test['label']
    return X_train, X_test, y_train, y_test

def padding_data( tokenizer, X_train, X_test):
    X_train_sequences = tokenizer.texts_to_sequences(X_train)
    X_test_sequences = tokenizer.texts_to_sequences(X_test)
    # Max Len in X_train_sequences
    maxlen = max(len(tokens) for tokens in X_train_sequences)
    # Perform padding on X_train and X_test sequences
    X_train_padded = pad_sequences(X_train_sequences, maxlen=maxlen, padding='post')
    X_test_padded = pad_sequences(X_test_sequences, maxlen=maxlen, padding='post')
    # Embedding Vocabulary Size 
    vocabulary_size = len(set(token for sequence in X_train_padded for token in sequence))
    return X_train_padded, X_test_padded, vocabulary_size

# model building
def model_build( X_train_padded, X_test_padded, y_train, y_test,
                 vocabulary_size, GRU_layer, dropout, snd_layer_unit, trd_layer_unit,
                 epochs,batch_size):
    # Define the model
    model = Sequential()
    # Add an embedding layer with input_dim=1000, output_dim=100, input_length=75
    model.add(Embedding(input_dim=vocabulary_size, output_dim=100))
    # Add a bidirectional GRU layer with 128 units
    model.add(Bidirectional(GRU(GRU_layer)))
    # Add batch normalization layer
    model.add(BatchNormalization())
    # Add dropout regularization
    model.add(Dropout(dropout))
    # Add a dense layer with 64 units and ReLU activation
    model.add(Dense(snd_layer_unit, activation='relu'))
    # Add dropout regularization
    model.add(Dropout(dropout))
    # Add the output layer with 6 units for 6 labels and softmax activation
    model.add(Dense(trd_layer_unit, activation='softmax'))
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Model Train 
    history = model.fit(X_train_padded, y_train,
                     epochs=epochs, batch_size=batch_size,
                       validation_data=(X_test_padded, y_test))
    return model

# Save the model
# Save the model
'''
def save_model(model, architecture_path, weights_path):
    # Save model architecture to JSON
    model_architecture = model.to_json()
    with open(architecture_path+'/architecture.json', 'w') as json_file:
        json_file.write(model_architecture)
    # Save model weights to HDF5
    model.save_weights(weights_path+'/model.weights.h5')
    '''
def save_model(model, weights_path):
    model.save(weights_path+'/model.h5')


def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir.as_posix()+'/params.yaml'
    params = yaml.safe_load(open(params_file))["train_model"]

    input_file_train = sys.argv[1]
    input_file_test = sys.argv[2]
    data_path_train = home_dir.as_posix() + input_file_train
    data_path_test = home_dir.as_posix() + input_file_test
    #arch_path = home_dir.as_posix() + '/models'
    #pathlib.Path(arch_path).mkdir(parents=True, exist_ok=True)
    weight_path = home_dir.as_posix() + '/models'
    pathlib.Path(weight_path).mkdir(parents=True, exist_ok=True)
    token_path = home_dir.as_posix() + '/data/interim'
    tokenizer = load_tokenizer( token_path + '/tokenizer.json')
    X_train, X_test, y_train, y_test= load_and_split_data(data_path_train,
                                                           data_path_test)
    X_train_padded, X_test_padded, vocabulary_size=padding_data(tokenizer,
                                                                X_train,
                                                                 X_test)
    model = model_build( X_train_padded, X_test_padded,
                         y_train, y_test, vocabulary_size,
                         params['GRU_layer'], params['dropout'], 
                         params['snd_layer_unit'], params['trd_layer_unit'],
                         params['epochs'],params['batch_size'])
    
    save_model(model, weight_path)

if __name__=="__main__":
    main()   






