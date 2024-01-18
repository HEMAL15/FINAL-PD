# app.py

from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load features (columns excluding 'plugin', 'codec', 'level')
df = pd.read_csv('COMP.csv', encoding='ISO-8859-1')
features = df.columns.difference(['plugin', 'codec', 'level'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommendation.html', methods=['GET', 'POST'])
def recommendation():
    if request.method == 'POST':
        return render_template('results.html', user_input={})

    return render_template('recommendation.html')

@app.route('/results.html', methods=['GET', 'POST'])
def results():
    if request.method == 'POST':
        # Load models from pickle files
        with open('model_plugin.pkl', 'rb') as model_file:
            model_plugin = pickle.load(model_file)

        with open('model_codec.pkl', 'rb') as model_file:
            model_codec = pickle.load(model_file)

        with open('model_level.pkl', 'rb') as model_file:
            model_level = pickle.load(model_file)

        # Get user input from the form
        user_input = request.form

        # Create a DataFrame with user input
        user_df = pd.DataFrame([user_input], columns=features)

        # Make predictions for the user input
        predicted_plugin = model_plugin.predict(user_df)
        predicted_codec = model_codec.predict(user_df)
        predicted_level = model_level.predict(user_df)

        return render_template('results.html', 
                               user_input=user_input,
                               predicted_plugin=predicted_plugin[0],
                               predicted_codec=predicted_codec[0],
                               predicted_level=predicted_level[0])

    return render_template('results.html')

if __name__ == '__main__':
    app.run(debug=True)
