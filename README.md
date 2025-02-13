
# 😊 Sentiment Analysis Project with NLTK

## 🚀 Overview
This project performs sentiment analysis using the Natural Language Toolkit (NLTK) in Python. It classifies text as positive, negative, or neutral based on sentiment scores. The project demonstrates preprocessing, tokenization, and sentiment analysis techniques using NLTK's built-in tools.

## ✨ Features
- ✅ Text preprocessing (tokenization, stopword removal, and stemming/lemmatization)
- ✅ Sentiment classification using NLTK's Vader Sentiment Analyzer
- ✅ Visualization of sentiment distribution
- ✅ User input for real-time sentiment analysis

## 📂 Project Structure
```
📦Sentiment-Analysis
 ┣ 📂templates
 ┃ ┗ 📜index.html
 ┣ 📜Sentiment Analysis Project with NLTK.ipynb
 ┣ 📜app.py
 ┣ 📜clf.pkl
 ┣ 📜requirements.txt
 ┗ 📜tfidf.pkl
```

- `Sentiment Analysis Project with NLTK.ipynb`: Jupyter Notebook demonstrating the sentiment analysis process.
- `app.py`: Flask web application for real-time sentiment analysis.
- `clf.pkl`: Serialized classifier model.
- `tfidf.pkl`: Serialized TF-IDF vectorizer.
- `requirements.txt`: List of required Python libraries.
- `templates/index.html`: HTML template for the Flask app.

## 📚 Libraries Used
- **Python 3.7+**
- **NLTK**: Natural Language Toolkit for text processing.
- **Flask**: Web framework for the application.
- **Pickle**: For serializing the model and vectorizer.
- **Jupyter Notebook**: For interactive code demonstrations.

## 📥 Installation
### Prerequisites
Ensure you have Python 3.7 or higher installed.

### Install Dependencies
Navigate to the project directory and run:
```bash
pip install -r requirements.txt
```

## 🛠️ Usage
1. **Run the Flask App**:
   ```bash
   python app.py
   ```
   Access the web application at `http://127.0.0.1:5000/`.

2. **Run the Jupyter Notebook**:
   Open `Sentiment Analysis Project with NLTK.ipynb` to explore the sentiment analysis process step-by-step.

## 🤝 Contributing
Feel free to fork this repository and submit pull requests with improvements or additional features.

## 📜 License
This project is licensed under the MIT License.

## 👨‍💻 Author
[Pinju Patel](https://github.com/PinjuPatel13)
