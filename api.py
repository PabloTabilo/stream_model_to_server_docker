from flask import Flask, jsonify, request
import numpy as np
import pickle

app = Flask(__name__)

FILENAME = "logit_marks_admissions.sav"
THRESHOLD = 0.5

class Model:
    def __init__(self, path):
        self.path = path
    def load(self):
        return pickle.load(open(self.path,"rb"))

logit = Model(FILENAME)
logit_p = logit.load()

def process_data(data):
    return np.array([[np.float32(data["mark1"]),np.float32(data["mark2"])]])

@app.route("/")
def call_server_on():
    return jsonify({"message":"The server is ON!"})

@app.route("/res", methods = ["POST"])
def scoring():
    response = {
                "score" : 0,
                "decision" : 0
            }
    if(request.method == "POST"):
        vec = process_data(request.get_json())
        response["score"] = logit_p.predict_proba(vec)[0][0]
        response["decision"] = 1 if response["score"] >= THRESHOLD else 0
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5001)
