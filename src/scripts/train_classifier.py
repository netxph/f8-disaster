import sys, os, pathlib
import pandas as pd
import re
import joblib

from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# cache these to speed up tokenization
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words("english")
pattern = re.compile(r"[^a-zA-Z0-9]")

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


def tokenize(text):
    """Tokenizes the text

    Args:
        text (str): The text to tokenize

    Returns:
        tokens (list): The list of tokens
    """

    text = pattern.sub(" ", text.lower())
    tokens = word_tokenize(text)

    tokens = [lemmatizer.lemmatize(token).strip() for token in tokens if token not in stop_words]

    return tokens

def convert_params(params):
    """Convert GridSearchCV best parameters to acceptable dictionary format that can be fed again to GridSearchCV.

    Args:
        params (dict): Dictionary of best parameters.

    Returns:
        dict: Dictionary of best parameters in acceptable format.
    """
 
    dict = {}
    for key in params:
        dict[key] = [params[key]]

    return dict

def get_metrics(y_test, y_pred):
    """Flatten classification report dictionary to dataframe for easier processing.
    
    Args:
        y_test (pandas.DataFrame): Test set labels.
        y_pred (pandas.DataFrame): Predicted labels.

    Returns:
        pandas.DataFrame: Classification report in dataframe format.
    """

    scores = []

    for i, col in enumerate(y_test.columns):
        report = classification_report(y_test.iloc[:, i], y_pred[:, i], output_dict=True, zero_division=0)

        scores.append({ 
            "category": col,
            "precision_0": report["0"]["precision"],
            "recall_0": report["0"]["recall"],
            "f1_0": report["0"]["f1-score"],
            "support_0": report["0"]["support"],
            "precision_1": report["1"]["precision"],
            "recall_1": report["1"]["recall"],
            "f1_1": report["1"]["f1-score"],
            "support_1": report["1"]["support"],
            "accuracy": report["accuracy"],
            "precision_macro_avg": report["macro avg"]["precision"],
            "recall_macro_avg": report["macro avg"]["recall"],
            "f1_macro_avg": report["macro avg"]["f1-score"],
            "support_macro_avg": report["macro avg"]["support"],
            "precision_weighted_avg": report["weighted avg"]["precision"],
            "recall_weighted_avg": report["weighted avg"]["recall"],
            "f1_weighted_avg": report["weighted avg"]["f1-score"],
            "support_weighted_avg": report["weighted avg"]["support"]
        })

    return pd.DataFrame.from_records(scores)

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

    y_pred = model.predict(X_test)
    metrics = get_metrics(Y_test, y_pred)

    print(metrics.mean(numeric_only=True))


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)

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