from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/findthedonor', methods=["GET", "POST"])
def findthedonor():
    if request.method == "POST":
        input_features = [float(request.form[x]) for x in ['recency', 'frequency', 'monetary', 'time']]
        features = [np.array(input_features)]
        prediction = model.predict(features)
        
        col = ['recency', 'frequency', 'monetary', 'time']
        df = pd.DataFrame(features, columns=col)
        
        output = prediction
        return render_template('findthedonor.html', prediction_text='Chance of donor to donate blood is {}'.format(output))

    # If it's a GET request or initial load, render the form
    return render_template('findthedonor.html', prediction_text='')

if __name__ == "__main__":
    app.run(debug=True, port=5000)