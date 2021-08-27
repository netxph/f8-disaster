import pandas as pd
import os
from flask import Blueprint
from sqlalchemy import create_engine
from pathlib import Path
from src.plots import Plots

api_bp = Blueprint('api', __name__)

@api_bp.route("/graph/categories")
def graph_categories():
    path = os.path.abspath(f"{os.path.dirname(os.path.abspath(__file__))}/../../data/processed/DisasterResponse.db")
    path = f"sqlite:///{Path(path).as_posix()}" 

    print(path)
    df = pd.read_sql_table("Message", create_engine(path))
    plots = Plots(df)
    
    return plots.get_categories()
