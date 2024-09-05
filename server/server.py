from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from joblib import load
import stanza
import re
import nltk
from nltk.corpus import stopwords
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the model and LabelEncoder
model = load('model/emotion_model.joblib')
le = load('model/label_encoder.joblib')

# Download and initialize Stanza
stanza.download('hebrew')
nlp = stanza.Pipeline('hebrew', processors='tokenize,mwt,pos,lemma') #(Multi-Word Token expansion): This processor handles the identification and expansion of multi-word expressions (MWEs). MWEs are phrases that convey a single meaning (e.g., "New York" or "חיים טובים" meaning "good life"). The mwt processor recognizes these phrases as single tokens during tokenization.

# Hebrew stopwords
nltk.download('stopwords')
hebrew_stopwords = set(stopwords.words('hebrew'))

# Function to preprocess text using Stanza
def preprocess_text(text):
    # Tokenize and lemmatize using Stanza
    doc = nlp(text)
    tokens = [word.lemma.lower() for sent in doc.sentences for word in sent.words]
    
    # Remove non-Hebrew characters and stopwords
    cleaned_tokens = [word for word in tokens if re.match('[א-ת]', word) and (word not in hebrew_stopwords or word == 'לא')]
    
    return ' '.join(cleaned_tokens)

def chunk_text(text, max_chunk_size=100):
    #"""Splits text into smaller chunks."""
    # Use punctuation to split the text
    sentences = re.split(r'[.?!;]', text)
    chunks = []
    
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def predict_emotion_for_long_text(text, model, aggregate_func=max):
    #"""Predicts the emotion for a long text by chunking and aggregating."""
    # Chunk the text
    chunks = chunk_text(text)
    predictions = []
    
    for chunk in chunks:
        processed_chunk = preprocess_text(chunk)
        predicted_emotion = model.predict([processed_chunk])[0]
        predictions.append(predicted_emotion)
    
    # Aggregate results with a more sophisticated method
    prediction_probabilities = model.predict_proba([preprocess_text(text)])[0]
    aggregated_prediction = np.argmax(prediction_probabilities)#finds the index of the maximum value in predict_propabilities
    #,determines wich emotion is most likely for the whole text,based on the chucks emotions we already calculated
    
    return aggregated_prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    processed_text = preprocess_text(text)
    predicted_emotion = predict_emotion_for_long_text(processed_text, model, aggregate_func=max)
    predicted_emotion_label = le.inverse_transform([predicted_emotion])[0] 
    return jsonify({'emotion': predicted_emotion_label})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

