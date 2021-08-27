import sys
import joblib
import pandas as pd

def load_model(model_filepath):
    """Loads a pickled model from a filepath

    Args:
        model_filepath (str): The filepath to the model

    Returns:
        model (sklearn.pipeline.Pipeline): The loaded model
    """
    return joblib.load(model_filepath)

def predict(model, message):
    """Predicts the category of a message

    Args:
        model (sklearn.pipeline.Pipeline): The model to use
        message (str): The message to predict

    Returns:
        categories (list): The predicted categories 
    """

    df = pd.DataFrame({ "message": [message], "genre":["direct"] })
    
    y = model.predict(df)
    categories = pd.DataFrame(y, columns=model.feature_names)

    categories = categories.stack().reset_index().rename(columns={"level_1": "category", 0: "value"})[["category", "value"]]

    return categories[categories["value"] == 1]

def main():
    if len(sys.argv) == 3:
        model_filepath, message = sys.argv[1:]

        model = load_model(model_filepath)

        y = predict(model, message)

        print(y)

if __name__ == '__main__':
    main()