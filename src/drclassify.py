import pandas as pd

class DRClassify:
    """Disaster Recovery classification class"""

    def __init__(self, model):
        """Initializes DRClassify class

        Args:
            model (sklearn.pipeline.Pipeline): DR classification model
        """

        self._model = model

    def classify(self, message):
        """Classifies a message

        Args:
            message (str): Message to classify
        Returns:
            categories (pd.DataFrame): Categories classification result
        """

        X = pd.DataFrame({ "message": [message], "genre":["direct"] })
        y = self._model.predict(X)

        categories = pd.DataFrame(y, columns=self._model.feature_names)
        
        categories = categories.stack().reset_index().rename(columns={"level_1": "category", 0: "value"})[["category", "value"]]

        return categories