from flask import Flask, request, json, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('titanic_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    
    features = [float(x) for x in request.form.values()]
    features = np.array(features).reshape(1, -1)
    
    
    features = scaler.transform(features)
    
    prediction = model.predict(features)
    
    return render_template('index.html', prediction_text='Survival Prediction: {}'.format('Survived' if prediction[0] == 1 else 'Not Survived'))

if __name__ == '__main__':
    app.run(debug=True, port= 5001)