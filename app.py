# text preprocessing modules
from string import punctuation 
# text preprocessing modules
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  # regular expression
import os
import pathlib
from os.path import dirname, join, realpath
import joblib
import uvicorn
from fastapi import FastAPI
from tensorflow.keras.models import model_from_json
# Download nltk resources
nltk.download('punkt')
nltk.download('stopwords') 
nltk.download('wordnet')

# Initialize a FastAPI App Instance
app = FastAPI(
    title="Sentiment Model API",
    description="A simple API that use NLP model to predict the sentiment of the sentences",
    version="0.1",
)
# Load model
curr_path = pathlib.Path(__file__).parent.absolute()
with open(os.path.join(curr_path, 'models', 'architecture.json'), 'r') as json_file:
    loaded_model_architecture = json_file.read()
loaded_model = model_from_json(loaded_model_architecture)
loaded_model.load_weights(os.path.join(curr_path, 'models', 'model.weights.h5'))

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Define function to perform text cleaning
def text_cleaning(text):
    # Remove URLs
    text = re.sub(r'https?\S+', '', text)
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove numeric values
    text = re.sub(r'\d+', '', text)
    # Lowercase the text
    text = text.lower()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Lemmatize text
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())
    return text

# Create prediction endpoint
@app.get("/predict-sentiment")
def predict_sentiment(review: str):
    # Clean the review
    cleaned_review = text_cleaning(review)
    # Perform prediction
    prediction = loaded_model.predict([cleaned_review])
    output = int(prediction[0])
    probabilities = loaded_model.predict_proba([cleaned_review])
    output_probability = "{:.2f}".format(float(probabilities[:, output]))
    # Map predicted sentiment to corresponding label
    sentiments = {0: "Sadness", 1: "Joy", 2: "Love", 3: 'Anger', 4: 'Surprise'}
    # Format prediction result
    result = {"prediction": sentiments[output], "probability": output_probability}
    return result

if __name__=="__main__":
    import uvicorn 
    uvicorn.run(app, host="0.0.0.0", port= 8088)