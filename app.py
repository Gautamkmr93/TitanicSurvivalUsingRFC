import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('trainingpklmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0])

    return render_template('index.html', prediction_text='this person get score {}  , \n meaning of 1 is survive meaning of 0 is not survive'.format(output))

#@app.route('/predict_api',methods=['POST'])
#def predict_api():
#    '''
#    For direct API calls trought request
#    '''
#    data = request.get_json(force=True)
#    prediction = model.predict([np.array(list(data.values()))])

#    output = prediction[0]
#    return jsonify(output)'''

if __name__ == "__main__":
    app.run(port=5000,debug=True)