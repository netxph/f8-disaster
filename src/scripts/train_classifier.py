import sys, os, pathlib
import pandas as pd
import joblib
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from src.utils import tokenize, get_metrics

def load_data(database_filepath):
    """Loads the data from the database and returns the dataframe

    Args:
        database_filepath (str): The path to the database file

    Returns:
        df (pandas.DataFrame): The dataframe containing the data
    """

    path = pathlib.Path(os.path.abspath(database_filepath)).as_uri().replace("file:", "sqlite:")
    engine = create_engine(path)

    df = pd.read_sql("SELECT * FROM Message", engine)

    GenreType = pd.CategoricalDtype(categories=["news", "direct", "social"], ordered=False)
    df.genre = df.genre.astype(GenreType)

    X = df[["message", "genre"]]
    y = df.drop(columns=["id", "message", "genre", "original"])
    category_names = y.columns

    return X, y, category_names

def build_model():
    """Creates a GridSearchCV pipeline for the model.

    Parameters include unigram and bigram, minimum and maximum document frequency, and the classifier.

    Args:
        None

    Returns:
        sklearn.pipeline.Pipeline: The GridSearchCV pipeline.
    """

    pipeline = Pipeline([
        ("features", ColumnTransformer([
            ("genre_category", OneHotEncoder(dtype=int), ["genre"]),
            ("text", Pipeline([
                ("count", CountVectorizer(tokenizer=tokenize)),
                ("tfidf", TfidfTransformer())
            ]), "message")
        ], remainder="drop")),
        ("clf", MultiOutputClassifier(MultinomialNB()))
    ])

    parameters = {
        "features__text__count__ngram_range": [(1, 1), (1, 2)],
        "features__text__count__min_df": [1, 10, 20],
        "clf__estimator": [
            LogisticRegression(max_iter=1000),
            LinearSVC(),
            MultinomialNB()]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=10, n_jobs=8)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluates the model on the test set.

    Args:
        model (sklearn.pipeline.Pipeline): The model to evaluate.
        X_test (pandas.DataFrame): The test set.
        Y_test (pandas.DataFrame): The test labels.
        category_names (list): The category names.

    Returns:
        None
    """

    # monkey patch category names

    model.feature_names = category_names
    y_pred = model.predict(X_test)
    metrics = get_metrics(Y_test, y_pred)

    print(metrics[["category", "f1_macro_avg", "precision_macro_avg", "recall_macro_avg", "accuracy"]])


def save_model(model, model_filepath):

    path = os.path.abspath(model_filepath)
    pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        print(model.best_params_)
        
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