import re
import os
import pandas as pd
import numpy as np
from pathlib import Path
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report

# cache these to speed up tokenization
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words("english")

pattern = re.compile(r"[^a-zA-Z0-9]")

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

def get_path(current, path):
    path = os.path.abspath(f"{os.path.dirname(os.path.abspath(current))}/{path}")

    return Path(path).as_posix()