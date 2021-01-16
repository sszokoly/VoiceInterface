
# Voice Interface Demo

## Intro

Voice interface equipped systems expect keywords from the user to control the application.
The purpose of this demo is to build a simple voice interface which can distinguish 10 words and detect
which one was spoken by the user in a wav file.

## Problem

Computers cannot recognize words. They need to be taught to tell a word apart from another.

## Solution

Build and train a CNN model which can detect (predict) the keyword received from the user. 

## How it works

Extracts the Mel-Frequency Cepstral coefficients from the audio files of the source dataset. Builds a computer neural network (CNN) with Tensorflow/Keras and feeds the MFCCs through the network in the training process. Provides a HTTP service interface
where the user (HTTP client) can POST a wav file and the service (HTTP server) returns the keyword predicted by the model
in JSON format. 

## Disclaimer

The author in no way provides any warranty, express or implied, towards the content of this tool. Use at your
own discretion and risk.