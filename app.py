# text preprocessing modules
from string import punctuation 
# text preprocessing modules
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  # regular expression
import pathlib
from os.path import dirname, join, realpath
from fastapi import FastAPI, Request
from tensorflow.keras.models import model_from_json
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
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

# Load the pre-trained sentiment analysis model
model_path = pathlib.Path(__file__).parent / "models/model.keras"
print(model_path)
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
    return text

# Create prediction endpoint
@app.get("/predict-sentiment/")
async def predict_sentiment(request: Request):
    return templates.TemplateResponse("predict_sentiment.html", {"request": request})

@app.post("/predict-sentiment/result/")
async def predict_sentiment_result(request: Request, review: str):
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
    return templates.TemplateResponse("predict_result.html", {"request": request, "result": result})

if __name__ == "__main__":
    import uvicorn 
    uvicorn.run(app, host="0.0.0.0", port=8088)