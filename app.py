import pickle

from flask import Flask,render_template,request
import numpy as np
import xgboost
import sklearn

app = Flask(__name__)

model = pickle.load(open('/Users/mitanshpatel/Downloads/calpred.pkl','rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict_cals():
    int_features = [int(x) for x in request.form.values()]
    features = [np.array(int_features)]
    prediction = model.predict(features)

    return str(prediction)


if __name__ == '__main__':
 app.run(host="0.0.0.0",port=8003)
