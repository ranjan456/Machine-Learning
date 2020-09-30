import numpy as np
from flask import Flask, request, jsonify, render_template
import sklearn.externals
import joblib

app = Flask(__name__)
model = joblib.load(open('multiple_linear_regression.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('pred.html')

@app.route('/predict',methods=['POST'])
def predict():

    NewYork = float(request.form['New York'])
    California = float(request.form['California'])
    Florida = float(request.form['Florida'])
    RnD_Spend = float(request.form['RnD_Spend'])
    Administration = float(request.form['Administration'])
    Marketing_Spend = float(request.form['Marketing_Spend'])
    pred_args = [NewYork,California,Florida,RnD_Spend,Administration,Marketing_Spend]
    pred_args_arr = np.array(pred_args)
    pred_args_arr = pred_args_arr.reshape(1,-1)

    model_prediction = model.predict(pred_args_arr)
    model_prediction = round(float(model_prediction),2)
    return render_template('pred.html', prediction='Prediction $ {}'.format(model_prediction))


if __name__ == "__main__":
    app.run(debug=True)
