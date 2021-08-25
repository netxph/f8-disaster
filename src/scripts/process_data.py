import sys, os, pathlib
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
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
    # drop columns that has only one value
    one_unique_cols = df.loc[:, df.nunique() == 1].columns.to_list()
    df = df.drop(columns=one_unique_cols)

    # drop duplicates
    df = df.drop_duplicates(subset=["id"])
    df = df[df.message != "#NAME?"]

def save_data(df, database_filename):
    engine = create_engine(pathlib.Path(os.path.abspath(database_filename)).as_uri())
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
