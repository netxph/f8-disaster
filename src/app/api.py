import pandas as pd
from flask import Blueprint, request
from sqlalchemy import create_engine
from src.plots import Plots
from src.utils import get_path
from src.drclassify import DRClassify
import joblib

api_bp = Blueprint('api', __name__)

path = get_path(__file__, "../../data/processed/DisasterResponse.db")
path = f"sqlite:///{path}" 

df = pd.read_sql_table("Message", create_engine(path))
plots = Plots(df)

model = joblib.load(get_path(__file__, "../../models/disaster_response_model.pkl"))
clf = DRClassify(model)

@api_bp.route("/graph/categories")
def graph_categories():

    return plots.get_categories()

@api_bp.route("/graph/words")
def graph_words():
    
    return plots.get_words()

@api_bp.route("/messages/classify")
def classify_message():
    message = request.args.get("message")
    categories = clf.classify(message)

    return categories.to_json(orient="records")