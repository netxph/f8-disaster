import json
import pandas as pd
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
from src.utils import tokenize

class Plots:
    """Process data for graph plots"""

    def __init__(self, data):
        """Initialize Plots object

        Args:
            data (pandas.DataFrame): DataFrame of disaster response messages
        """
        self.data = data
        self._categories_json = ""
        self._words_json = ""

    def get_categories(self):
        """Get message categories distribution

        Returns:
            json: json of plotly figure
        """
        if self._categories_json =="":
            categories = pd.melt(self.data.drop(columns=['id', 'message', 'original']).groupby("genre").sum().reset_index(), id_vars=['genre'], var_name="category", value_name="count")

            fig = px.bar(categories, x="category", y="count", color="genre", title="Distribution of Message Categories")

            self._categories_json = json.dumps(fig, cls=PlotlyJSONEncoder)

        return self._categories_json

    def get_words(self):
        """Get top 20 word counts

        Returns:
            json: json of plotly figure
        """

        if self._words_json == "":
            tokens = self.data.message.apply(lambda text: tokenize(text)).explode().value_counts().reset_index().rename(columns={'index': 'token', 'message': 'count'})

            fig  = px.bar(tokens.head(20), x="token", y="count", title="Top 20 Words")

            self._words_json = json.dumps(fig, cls=PlotlyJSONEncoder)

        return self._words_json