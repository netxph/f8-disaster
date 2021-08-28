import sys, os, pathlib
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load data from csv files and merge them into one dataframe

    Args:
        messages_filepath (str): path to messages csv file
        categories_filepath (str): path to categories csv file

    Returns:
        df (pandas.DataFrame): merged dataframe
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = pd.merge(messages, categories, on='id')

    categories = df.categories.str.split(";", expand=True)
    
    row = categories.iloc[0]
    row = row.apply(lambda col: col[:-2])
    category_colnames = row.to_list()

    categories.columns = category_colnames

    for column in categories:
    # we need to ensure that no other values than zeroes and ones
        categories[column] = categories[column].apply(lambda col: col[-1]).apply(lambda col: "0" if col == "0" else "1") 
    
        categories[column] = pd.to_numeric(categories[column])

    df = df.drop(columns=["categories"])
    df = pd.concat([df, categories], axis=1)

    return df

def clean_data(df):
    """Clean dataframe by removing duplicates and dropping columns with only one value

    Args:
        df (pandas.DataFrame): dataframe to clean

    Returns:
        df (pandas.DataFrame): cleaned dataframe

    """


    # drop columns that has only one value
    one_unique_cols = df.loc[:, df.nunique() == 1].columns.to_list()
    df = df.drop(columns=one_unique_cols)

    # drop duplicates
    df = df.drop_duplicates(subset=["id"])
    df = df[df.message != "#NAME?"]

    return df

def save_data(df, database_filename):
    """Save dataframe to sqlite database

    Args:
        df (pandas.DataFrame): dataframe to save
        database_filename (str): path to database file. Path should be in a directory format not SQLite URI format.

    Returns:
        None
    """
    path = os.path.abspath(database_filename)
    pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)

    path = pathlib.Path(path).as_uri().replace("file:", "sqlite:")
    engine = create_engine(path)
    df.to_sql('Message', engine, if_exists="replace", index=False)


def main():
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
