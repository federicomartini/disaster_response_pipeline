import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import pandas as pd
import sqlite3
from sqlalchemy import create_engine
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
import pickle

def load_data(database_filepath):
    """Load data from a SQLite Database to create the DataFrames about messages, categories and a series containing the category columns 
    
    Arguments:
        database_filepath : String
            The location of the SQLite Database
 
    Output:
        X : DataFrame
            The Pandas DataFrame containing the messages
        Y : DataFrame
            The Pandas DataFrame containing the categories
        categories : Pandas Series
            The Pandas Series containing the column names of the categories
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("disasterResponse", engine)
    X = df['message']
    Y = df.drop(['message', 'id', 'original', 'genre'], axis=1)
    categories = Y.columns
    
    return X, Y, categories

def tokenize(text):
    """Tokenize function 
    
    Arguments:
        text : String
            Text to tokenize
 
    Output:
        clean_tokens : List
            List of clean tokens after cleaning, word-tokenizing and Lemmatizing the text
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []

    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """Build the model 
 
    Output:
        model : Model
            The model created
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [10],
        'clf__estimator__min_samples_split': [2, 4],
    }
    
    model = GridSearchCV(pipeline, param_grid=parameters, verbose=4, n_jobs=-1)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """Model evaluation function 
    
    Arguments:
        model : Model
            The Model to evaluate
        X_test : Pandas DataFrame
            The DataFrame containing the features
        Y_test : Pandas DataFrame
            The DataFrame containing the labels
        category_names : List of Strings
            The list of the category names
            
    Output:
        None
    """
    y_pred = model.predict(X_test)
    
    for column in category_names:
        print(classification_report(Y_test[column], pd.DataFrame(y_pred, columns = category_names)[column]))


def save_model(model, model_filepath):
    """Save the model into a file 
    
    Arguments:
        model : Model
            The Model to evaluate
        model_filepath : String
            The location of the model file to save

    Output:
        None
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    """Function to execute the whole train process 
    
    Steps:
        1) Load data from the SQLite Database
        2) Train the model using the GridSearchCV
        3) Test and Evaluate the model 
        4) Save the model into a file

    Output:
        None
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()