# -*- coding: utf-8 -*-
"""
Keyword detecting service for HTTP server.
"""

import librosa
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf

MODEL_PATH = "model.h5"
SAMPLES_TO_CONSIDER = 22050 # 1 sec


class _Keyword_Spotting_Service:

    model = None
    mappings = [     
        "down",
        "go",
        "left",
        "no",
        "off",
        "on",
        "right",
        "stop",
        "up",
        "yes"
    ]
    _instance = None

    def predict(self, file_path):

        # extract MFCCs (# segments x # coefficients)
        MFCCs = self.preprocess(file_path)

        # convert 2D MFCCs into 4D array, (#samples x #segments x #coefficients x #channels)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # make prediction  
        # MFCCs = tf.convert_to_tensor(MFCCs, np.float32)
        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        predicted_keyword = self.mappings[predicted_index]
        return predicted_keyword

    def preprocess(self, file_path, n_mfcc=13, hop_length=512, n_fft=2048):

        # load audiofile
        signal, sr = librosa.load(file_path)

        # ensure consistency in the audio file length
        if len(signal) > SAMPLES_TO_CONSIDER:
            signal = signal[:SAMPLES_TO_CONSIDER]

        # extract MFCCs
        MFCCs = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)

        return MFCCs.T


def Keyword_Spotting_Service():
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service._instance.model = keras.models.load_model(MODEL_PATH)
    return _Keyword_Spotting_Service._instance


if __name__ == "__main__":
    kss = Keyword_Spotting_Service()
    keyword = kss.predict("test/test_right.wav")
    print(f"Predicted Keyword: {keyword}")
