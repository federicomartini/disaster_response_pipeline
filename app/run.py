import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """Tokenize function 
    
    Arguments:
        text : String
            Text to tokenize
 
    Output:
        clean_tokens : List
            List of clean tokens after word-tokenizing and Lemmatizing the text
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disasterResponse', engine)

# load model
model = joblib.load("../models/multioutclassifier.pickle")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """Index function 
    
    Arguments:
        None
 
    Output:
        render_template : Render Template
            Render web page with plotly graphs
    """
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    #Data for visuals - Top 5 Common categories
    top_category_names = df.drop(['id', 'message', 'original', 'genre', 'related'], axis=1).sum().sort_values(ascending=False)[0:5].index
    top_category_values = df.drop(['id', 'message', 'original', 'genre', 'related'], axis=1).sum().sort_values(ascending=False)[0:5].values
    
    #Data for visuals - Least 5 Common categories
    least_category_names = df.drop(['id', 'message', 'original', 'genre', 'related'], axis=1).sum().sort_values(ascending=False)[-5:].index
    least_category_values = df.drop(['id', 'message', 'original', 'genre', 'related'], axis=1).sum().sort_values(ascending=False)[-5:].values
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
            # GRAPH 1 - genre graph
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        # GRAPH 2 - Top 5 Categories
        {
            'data': [
                Bar(
                    x=top_category_names,
                    y=top_category_values
                )
            ],

            'layout': {
                'title': 'Top 5 Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        # GRAPH 3 - Least 5 Categories
        {
            'data': [
                Bar(
                    x=least_category_names,
                    y=least_category_values
                )
            ],

            'layout': {
                'title': 'Least 5 Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """Go function 
    
    Arguments:
        None
 
    Output:
        render_template : Render Template
            Render the go.html Please see that file. 
    """
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """Main function 
    
    Arguments:
        None
 
    Output:
        None 
    """
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()