#!/usr/bin/env python
# encoding: utf-8
import json
from flask import Flask, request
from flask_cors import CORS, cross_origin
import pickle

app = Flask(__name__)
CORS(app, support_credentials=True)

model = pickle.load(open('Final_Model.pkl', 'rb'))
cv = pickle.load(open('cv_transfrom.pkl', 'rb'))


@app.route('/CheckOffensive')
@cross_origin(supports_credentials=True)
def search():
    vect = cv.transform([request.args.get("text")]).toarray()
    res = model.predict(vect)
    if res[0] == 1:
        return json.dumps({"result": 1})
    return json.dumps({"result": 0})


if __name__ == '__main__':
    app.run()
