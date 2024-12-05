# codtech-task1
*Name* : A.K.Tahseen Fathima

Company : CODTECH IT SOLUTIONS

Domain : Artificial Intelligence 

Duration : Nov 5 - Dec 5

Intern ID : CT08DS9831

# Overview of the project
This project implements a sentiment analysis system to classify movie reviews as either positive or negative using a Naive Bayes classifier. The model is trained using the NLTK movie reviews dataset, which contains 1,000 movie reviews categorized as positive and negative. The system is capable of predicting the sentiment of new, unseen reviews based on the model's learned patterns from the training data.

# project : Sentiment analysis

*Objective*:

The main objective of this project is to:
Build a text classification model that can predict the sentiment of a given movie review.
Understand and implement the use of machine learning algorithms for natural language processing (NLP) tasks.
Leverage NLTK's movie reviews dataset and machine learning libraries to develop a sentiment classifier.


The project aims to:
Demonstrate the use of a Naive Bayes classifier for sentiment analysis.
Preprocess text data for use in machine learning models.
Evaluate the performance of the sentiment classifier.


*Key Activities*:

Data Collection:
The project uses the movie reviews dataset from NLTK, which contains 1,000 movie reviews categorized into positive and negative sentiments.

Text Preprocessing:
The reviews are converted into a matrix of token counts using CountVectorizer.
The top 2,000 most frequent words in the dataset are retained for feature extraction.

Model Training:
The dataset is split into training and testing sets using train_test_split.
A Multinomial Naive Bayes classifier (MultinomialNB) is trained using the training data.

Model Evaluation:
The model’s performance is evaluated using accuracy metrics and prediction outcomes.
The model is tested on new text reviews using the predict_sentiment() function.

Sentiment Prediction:
The sentiment of new movie reviews is predicted by transforming the input text into a feature vector using the trained vectorizer and passing it through the Naive Bayes model.

*Technologies Used*:

Python: Programming language used for implementation.

NLTK (Natural Language Toolkit): Used for accessing the movie reviews dataset and handling text data.

Pandas: Utilized for data manipulation and structuring the dataset into a DataFrame.

Scikit-learn: A machine learning library used for implementing the Naive Bayes classifier and other machine learning utilities such as CountVectorizer, train_test_split, and MultinomialNB.


The model is evaluated based on its ability to classify movie reviews into the correct sentiment category, either positive or negati
