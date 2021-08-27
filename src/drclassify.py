import pandas as pd

class DRClassify:

    def __init__(self, model):
        self._model = model

    def classify(self, message):
        X = pd.DataFrame({ "message": [message], "genre":["direct"] })
        y = self._model.predict(X)

        categories = pd.DataFrame(y, columns=self._model.feature_names)
        
        categories = categories.stack().reset_index().rename(columns={"level_1": "category", 0: "value"})[["category", "value"]]

        return categories