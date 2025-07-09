from flask import Flask, render_template, request
from datetime import datetime
from sklearn.linear_model import LogisticRegression
# Importing necessary libraries
import numpy as np
import pandas as pd

app = Flask(__name__)  # initilizing the Flask app

# 1. Load Iris dataset from URL
iris = pd.read_csv("iris.csv")

# 2. Features and Target
X = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris['species']
model = LogisticRegression(max_iter=200)
# 3. Fit the model
model.fit(X, y)


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/predict', methods=['POST'])
def predict():
    sepal_length = float(request.form.get('sl'))
    sepal_width = float(request.form.get('sw'))
    petal_length = float(request.form.get('pl'))
    petal_width = float(request.form.get('pw'))
    # 4. Predicting a sample
    predicted_species = model.predict(
        [[sepal_length, sepal_width, petal_length, petal_width]])
    return render_template("home.html", data=predicted_species[0])


@app.route('/about')
def about():
    a = 10
    b = 20  # Example variables
    c = a * b
    # Get current time in a readable format
    current_time = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
    year = datetime.now().strftime("%Y")
    return render_template("about.html", sum=str(c), time=current_time, year=year)


@app.route('/contact')
def contact():
    return render_template("contact.html")


if __name__ == '__main__':
    app.run(debug=True)
# This will run the Flask app on all available IP addresses on port 5000
