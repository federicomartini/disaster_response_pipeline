import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from pandas.io import sql

def load_data(messages_filepath, categories_filepath):
    """Load messages and categories the merge them into a DataFrame on the ID field
    
    Arguments:
        messages_filepath : String
            The messages CSV file location
        categories_filepath : String
            The categories CSV file location
    Output:
        df : DataFrame
            The Pandas DataFrame containing the merged messages and categories
    """
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages, categories, on='id')
    
    return df

def clean_data(df):
    """Clean the Pandas DataFrame
    
    Arguments:
        df : DataFrame
            The Pandas DataFrame with messages and categories
    Output:
        df : DataFrame
            The Pandas DataFrame cleaned
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";", expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x: x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)  
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df

def save_data(df, database_filename):
    """Save the Pandas DataFrame into a SQL Database
    
    Arguments:
        df : DataFrame
            The cleaned Pandas DataFrame with messages and categories
        database_filename : String
            The file name of the SQL Database to create 
    """
    
    engine = create_engine('sqlite:///DisasterResponse.db')
    table_name = 'disasterResponse'
    sql.execute('DROP TABLE IF EXISTS %s'%table_name, engine)
    df.to_sql(table_name, engine, index=False)


def main():
    """
    Main function to perform the ETL Process
    
    Steps:
        - Merge messages and categories from the CSV files into a Pandas DataFrame
        - Clean the Pandas DataFrame
        - Load data into a SQL Database
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()