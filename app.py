import pathlib
import re  # regular expression
import json
# text preprocessing modules
from string import punctuation 
import numpy as np
# text preprocessing modules
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
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
# load tokenizer
def load_tokenizer(home_path):
    with open(home_path, 'r') as f:
        tokenizer_json = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_json)
    return tokenizer
# Load the pre-trained sentiment analysis model
model_path = pathlib.Path(__file__).parent/"model.h5"
loaded_model = load_model(model_path)
    

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Initialize templates
templates = Jinja2Templates(directory="templates")

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
    # Tokenize the text
    tokens = text.split()
    # load tokenizer
    curr_path = pathlib.Path(__file__).parent
    tokenizer =  load_tokenizer(curr_path.as_posix()+'/tokenizer.json')
    text= tokenizer.texts_to_sequences([tokens])
    # Perform padding on X_train and X_test sequences
    text = pad_sequences(text, maxlen=79, padding='post')
    return text


# Create prediction endpoint
@app.get("/predict-sentiment/")
async def predict_sentiment(request: Request):
    return templates.TemplateResponse("predict-sentiment.html", {"request": request})

# Modify endpoint to accept form data
@app.post("/predict-sentiment/result/")
async def predict_sentiment_result(request: Request, review: str = Form(...)):
    # Clean the review
    cleaned_review = text_cleaning(review)
    print(cleaned_review)
    # Perform prediction
    prediction = loaded_model.predict([cleaned_review])
    output =  np.argmax(prediction, axis=1)
    #probabilities = loaded_model.predict_proba([cleaned_review])
    #output_probability = np.argmax(probabilities, axis=1)
    #output_probability = "{:.2f}".format(float(probabilities[:, output]))
    # Map predicted sentiment to corresponding label
    sentiments = {0: "Sadness", 1: "Joy", 2: "Love", 3: 'Anger', 4: 'Surprise'}
    # Format prediction result
    result = {"prediction": sentiments[output[0]], "probability": prediction[0][output][0]}
    return templates.TemplateResponse("predict-result.html", {"request": request, "result": result})
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)