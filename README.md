# 😊 Sentiment Analysis Project with NLTK

## 🚀 Overview
This project performs sentiment analysis using the Natural Language Toolkit (NLTK) in Python. It classifies text as positive, negative, or neutral based on sentiment scores. The project demonstrates preprocessing, tokenization, and sentiment analysis techniques using NLTK's built-in tools.

## ✨ Features
- ✅ Text preprocessing (tokenization, stopword removal, and stemming/lemmatization)
- ✅ Sentiment classification using NLTK's Vader Sentiment Analyzer
- ✅ Visualization of sentiment distribution
- ✅ User input for real-time sentiment analysis

## 📥 Installation
### 🔹 Prerequisites
Ensure you have Python installed (preferably 3.7+). You also need Jupyter Notebook if you want to run the `.ipynb` file interactively.

### 🔹 Install Dependencies
Run the following command to install the required libraries:
```bash
pip install nltk pandas matplotlib
```

## 🛠️ Usage
1. 📂 Clone or download the repository.
2. 📜 Open the Jupyter Notebook (`Sentiment Analysis Project with NLTK.ipynb`).
3. ▶️ Run the notebook step by step to:
   - 🔄 Load and preprocess text data.
   - 📊 Analyze sentiment using NLTK.
   - 📈 Visualize results.
4. 📝 Modify the input text in the notebook to analyze custom sentences.

## 🎯 Example
```python
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
text = "I love this product! It's amazing."
sentiment = sia.polarity_scores(text)
print(sentiment)
```

Output:
```json
{'neg': 0.0, 'neu': 0.254, 'pos': 0.746, 'compound': 0.8516}
```

## 📁 Project Structure
```
📂 Sentiment-Analysis-Project
│-- 📄 Sentiment Analysis Project with NLTK.ipynb
│-- 📄 README.md
```

## 🤝 Contributing
Feel free to fork this repository and submit pull requests with improvements or additional features.

## 📜 License
This project is licensed under the MIT License.

## 👨‍💻 Author
Pinju_Patel

