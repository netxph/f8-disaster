import pandas as pd
from flask import Blueprint
from sqlalchemy import create_engine
from src.plots import Plots
from src.utils import get_path

api_bp = Blueprint('api', __name__)

path = get_path(__file__, "../../data/processed/DisasterResponse.db")
path = f"sqlite:///{path}" 

df = pd.read_sql_table("Message", create_engine(path))

@api_bp.route("/graph/categories")
def graph_categories():

    plots = Plots(df)
    
    return plots.get_categories()
