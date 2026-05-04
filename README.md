# Movie-Review-Sentiment-Classifier
IMDB movie review sentiment classifier using LSTM (Embedding + LSTM + Dropout + Dense). Classifies reviews as Positive (👍) or Negative (👎). Achieves ~88% accuracy on test data. Built with TensorFlow/Keras.
# IMDB Sentiment Analysis – LSTM

## Overview
This project uses a **Deep Learning LSTM model** to classify IMDB movie reviews as **Positive** or **Negative**.  
The model learns the sequence of words and understands the context to determine the sentiment of the review.

## Dataset
- **Source:** IMDB Dataset (50,000 movie reviews)
- **Classes:** Positive (1) / Negative (0)
- **Balance:** 25,000 positive + 25,000 negative

## Model Architecture
Embedding(input_dim=10000, output_dim=128)
↓
LSTM(128)
↓
Dropout(0.5)
↓
Dense(1, activation='sigmoid')

## Results
| Metric | Value |
|--------|-------|
| Test Accuracy | ~88.1% |
| Epochs | 5 |
| Batch Size | 64 |

## Features
- Text preprocessing (lowercase, remove HTML tags, remove special characters)
- Tokenization with 10,000 most frequent words
- Padding sequences to max length of 200
- LSTM layer for sequence understanding
- Dropout layer to reduce overfitting
- Prediction function for new reviews

## Technologies Used
- Python
- TensorFlow / Keras
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Regex



### 1. Install requirements
```bash
pip install tensorflow pandas numpy matplotlib scikit-learn
