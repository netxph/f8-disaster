import json
import plotly
import pandas as pd
import joblib
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine
from src.utils import tokenize
from src.app.api import api_bp

app = Flask(__name__)
app.register_blueprint(api_bp, url_prefix='/api')

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()