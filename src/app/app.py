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

# load data
# engine = create_engine('sqlite:///../../data/processed/DisasterResponse.db')
# df = pd.read_sql_table('Message', engine)

# load model
# model = joblib.load("../../models/disaster_response_model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    return render_template('master.html')

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()