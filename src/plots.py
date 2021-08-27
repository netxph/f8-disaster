import json
import pandas as pd
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

class Plots:

    def __init__(self, data):
        self.data = data

    def get_categories(self):
        categories = pd.melt(self.data.drop(columns=['id', 'message', 'original']).groupby("genre").sum().reset_index(), id_vars=['genre'], var_name="category", value_name="count")

        fig = px.bar(categories, x="category", y="count", color="genre")

        return json.dumps(fig, cls=PlotlyJSONEncoder)