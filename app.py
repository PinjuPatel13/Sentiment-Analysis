from flask import Flask, render_template, request
import pickle
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from googletrans import Translator

# 🔽 Download necessary NLTK datasets (only needed once)
nltk.download('wordnet')
nltk.download('stopwords')

# 🔽 Initialize objects
lemmatizer = WordNetLemmatizer()
stopwords_set = set(stopwords.words('english'))
translator = Translator()

app = Flask(__name__)

# 🔽 Load the sentiment analysis model and TF-IDF vectorizer
with open('clf.pkl', 'rb') as f:
    clf = pickle.load(f)
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# 🔽 Preprocessing Function
def preprocess_text(text):
    text = re.sub('<[^>]*>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords_set]  
    return " ".join(words)

@app.route('/', methods=['GET', 'POST'])
def analyze_sentiment():
    sentiment = None
    translated_text = None
    original_text = ""
    confidence = 0.0  # Default confidence score
    positive_count = 0
    negative_count = 0
    neutral_count = 0

    if request.method == 'POST':
        comment = request.form.get('comment')
        
        if comment.strip():
            original_text = comment

            # 🔽 Translate to English
            try:
                translated_text = translator.translate(comment, dest='en').text
            except Exception:
                translated_text = "Translation Error"

            # 🔽 Preprocess text
            preprocessed_text = preprocess_text(translated_text)

            # 🔽 Predict Sentiment
            if preprocessed_text:
                comment_vector = tfidf.transform([preprocessed_text])
                prediction = clf.predict_proba(comment_vector)  # Get probabilities
                
                confidence = round(float(np.max(prediction) * 100), 2)  # Confidence %
                
                if np.argmax(prediction) == 1:  # Assuming 1 = Positive, 0 = Negative
                    sentiment = "positive"
                    positive_count += 1
                elif np.argmax(prediction) == 0:
                    sentiment = "negative"
                    negative_count += 1
                else:
                    sentiment = "neutral"
                    neutral_count += 1
            else:
                sentiment = "neutral"

    return render_template(
        'index.html',
        sentiment=sentiment,
        original_text=original_text,
        translated_text=translated_text,
        confidence=confidence,
        positive_count=positive_count,
        negative_count=negative_count,
        neutral_count=neutral_count
    )

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

