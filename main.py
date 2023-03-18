import pickle, sklearn
import numpy
import numpy as np
from flask import Flask, render_template, request
import pandas as p

app = Flask(__name__)
data = p.read_csv('CleansedData.csv')
pipe = pickle.load(open('RidgeModel.pkl', 'rb'))


@app.route('/')
def index():
    location = sorted(data['locality'].unique())
    return render_template('index.html', locations=location)


@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = float(request.form.get('bhk'))
    sf = float(request.form.get('squareFeet'))
    input = p.DataFrame([[location, sf, bhk]], columns=['locality', 'area', 'bhk'])
    prediction = pipe.predict(input)[0] * 100000
    return str(np.round_(prediction, 2))


if __name__ == '__main__':
    app.run(debug=True, port=5000)
