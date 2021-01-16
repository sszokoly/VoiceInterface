# -*- coding: utf-8 -*-
"""
Prepares keyword prediction data from a dataset of wav files.
Dataset Source: https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html
"""

import json
import librosa
import os

DATASET_PATH = "dataset"
JSON_PATH = "data.json"
SAMPLES_TO_CONSIDER = 22050


def prepare_dataset(dataset_path, json_path, n_mfcc=13, hop_length=512, n_fft=2048):

    # data dictionary
    data = {
        "mappings": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    # loop through all subdirs
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # make sure we are not in root
        if dirpath is not dataset_path:

            # update mapping
            category = dirpath.split("/")[-1]
            data["mappings"].append(category)
            print(f"Processing {category}")

            # loop through files
            for f in filenames:
                # get file path
                file_path = os.path.join(dirpath, f)

                # load audio file
                signal, sr = librosa.load(file_path)

                # ensure audio is at least 1 sec long
                if len(signal) >= SAMPLES_TO_CONSIDER:
                    signal = signal[:SAMPLES_TO_CONSIDER]

                    # extract MFCCs
                    MFCCs = librosa.feature.mfcc(
                        signal, n_mfcc=n_mfcc,
                        hop_length=hop_length,
                        n_fft=n_fft
                    )

                    # update data dictionary
                    data["labels"].append(i-1)
                    data["MFCCs"].append(MFCCs.T.tolist())
                    data["files"].append(file_path)
                    print(f"{file_path}: {i-1}")

    # store in json
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    data = prepare_dataset(DATASET_PATH, JSON_PATH)
