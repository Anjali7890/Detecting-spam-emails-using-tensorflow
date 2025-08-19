# Detecting-spam-emails-using-tensorflow
## This project implements a deep learning model to classify emails as Spam or Ham (Not Spam) using TensorFlow & LSTM.
## The dataset is preprocessed with text cleaning, stopword removal, tokenization, and padding before training an LSTM-based neural network.
## Summary:
This project builds a machine learning pipeline to classify emails as Spam or Ham. 
The dataset is preprocessed with punctuation removal, stopword filtering, and text 
tokenization, followed by dataset balancing to improve model performance. 

An LSTM-based neural network with word embeddings is implemented in TensorFlow/Keras 
to capture sequential text patterns. The model is trained and validated with 
early stopping and learning rate scheduling to prevent overfitting. 

Results show high accuracy on unseen test data, with clear visualizations of model 
performance, enabling robust spam detection.

# Features
### Preprocessing of raw emails (punctuation & stopword removal)
### Balancing of dataset to handle class imbalance
### WordCloud visualization for spam vs ham emails
#### Tokenization & padding for sequence modeling
### LSTM-based deep learning model for classification
### Early stopping & learning rate scheduling for optimal training
### Performance visualization with accuracy plots

# Tech Stack
### Python 3
### TensorFlow / Keras
### NLTK (Stopwords removal)
### WordCloud (Visualization)
### Seaborn / Matplotlib (EDA & plots)

# Results

## Balanced dataset accuracy: ~90–95%
## Model: Embedding + LSTM + Dense layers
## Output:
## 0 → Ham (Not Spam)
## 1 → Spam

