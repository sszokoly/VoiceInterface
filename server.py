# -*- coding: utf-8 -*-
"""
HTTP server which receives audio file from HTTP client and predicts keyword.
"""

from flask import Flask, request, jsonify
from keyword_spotting_service import Keyword_Spotting_Service
import os
import random

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():

    # get audio file and save it
    audio_file = request.files["file"]
    file_name = str(random.randint(10, 100000))
    audio_file.save(file_name)

    # invoke keyword spotting service
    kss = Keyword_Spotting_Service()

    # make a prediction
    predicted_keyword = kss.predict(file_name)

    # remove audio file
    os.remove(file_name)

    # send predicted keyword in json format
    data = {"keyword": predicted_keyword}
    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=False)