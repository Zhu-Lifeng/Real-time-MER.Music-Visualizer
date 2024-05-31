from flask import Flask, render_template, Response
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
import json




def APP_Creation():
    notes_midi = pd.read_csv('notes_midi.csv').to_numpy()
    pitch_88 = [27.50, 29.14, 30.87, 32.70, 34.65, 36.71, 38.89, 41.20, 43.65, 46.25, 49.00, 51.91,
                55.00, 58.27, 61.74, 65.41, 69.30, 73.42, 77.78, 82.41, 87.31, 92.50,
                98.00, 103.83, 110.00, 116.54, 123.47, 130.81, 138.59, 146.83, 155.56,
                164.81, 174.61, 185.00, 196.00, 207.65, 220.00, 233.08, 246.94, 261.63,
                277.18, 293.66, 311.13, 329.23, 349.23, 369.99, 392.00, 415.30, 440.00,
                466.16, 493.88, 523.25, 554.37, 587.33, 622.25, 659.26, 698.46, 739.99,
                783.99, 830.61, 880.00, 932.33, 987.77, 1046.50, 1108.73, 1174.66, 1244.51,
                1318.51, 1396.91, 1479.98, 1567.98, 1661.22, 1760.00, 1864.66, 1975.53,
                2093.00, 2217.46, 2349.32, 2489.02, 2637.02, 2793.83, 2959.96, 3135.96,
                3322.44, 3520.00, 3729.31, 3951.07, 4186.01]
    PITCH_88A = ['A0', 'A#0', 'B0',
                 'C1', 'C#1', 'D1', 'D#1', 'E1', 'F1', 'F#1', 'G1', 'G#1', 'A1', 'A#1', 'B1',
                 'C2', 'C#2', 'D2', 'D#2', 'E2', 'F2', 'F#2', 'G2', 'G#2', 'A2', 'A#2', 'B2',
                 'C3', 'C#3', 'D3', 'D#3', 'E3', 'F3', 'F#3', 'G3', 'G#3', 'A3', 'A#3', 'B3',
                 'C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4',
                 'C5', 'C#5', 'D5', 'D#5', 'E5', 'F5', 'F#5', 'G5', 'G#5', 'A5', 'A#5', 'B5',
                 'C6', 'C#6', 'D6', 'D#6', 'E6', 'F6', 'F#6', 'G6', 'G#6', 'A6', 'A#6', 'B6',
                 'C7', 'C#7', 'D7', 'D#7', 'E7', 'F7', 'F#7', 'G7', 'G#7', 'A7', 'A#7', 'B7',
                 'C8']
    PITCH_88B = ['A0', 'B-0', 'B0',
                 'C1', 'D-1', 'D1', 'E-1', 'E1', 'F1', 'G-1', 'G1', 'A-1', 'A1', 'B-1', 'B1',
                 'C2', 'D-2', 'D2', 'E-2', 'E2', 'F2', 'G-2', 'G2', 'A-2', 'A2', 'B-2', 'B2',
                 'C3', 'D-3', 'D3', 'E-3', 'E3', 'F3', 'G-3', 'G3', 'A-3', 'A3', 'B-3', 'B3',
                 'C4', 'D-4', 'D4', 'E-4', 'E4', 'F4', 'G-4', 'G4', 'A-4', 'A4', 'B-4', 'B4',
                 'C5', 'D-5', 'D5', 'E-5', 'E5', 'F5', 'G-5', 'G5', 'A-5', 'A5', 'B-5', 'B5',
                 'C6', 'D-6', 'D6', 'E-6', 'E6', 'F6', 'G-6', 'G6', 'A-6', 'A6', 'B-6', 'B6',
                 'C7', 'D-7', 'D7', 'E-7', 'E7', 'F7', 'G-7', 'G7', 'A-7', 'A7', 'B-7', 'B7',
                 'C8']

    Q = 96961 // (1970.75 * 12)  # the number of records per quarter



    app = Flask(__name__)

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/stream')
    def stream():
        def generator(notes_midi, Q):
            record = np.zeros(88)
            num = 0
            for t in range(notes_midi.shape[0]):
                tem = []
                for p in range(88):
                    if notes_midi[t, p] == 1:
                        if record[p] == 0:
                            record[p] = t
                    else:
                        if record[p] != 0:
                            Offset = int(record[p] // Q)
                            Offset = Offset / 12 if Offset % 12 == 0 else Fraction(Offset, 12)
                            Duration = int((t - record[p]) // Q)
                            Duration = Duration / 12 if Duration % 12 == 0 else Fraction(Duration, 12)
                            Pitch = PITCH_88A[p]
                            tem.append([Pitch, Offset, Duration])
                            record[p] = 0
                dict_T = {}
                for element in tem:
                    key = (element[1], element[2])
                    if key not in dict_T:
                        dict_T[key] = element[0]
                    else:
                        dict_T[key] += ("," + element[0])
                tem = [[value, key[0], key[1]] for key, value in dict_T.items()]
                if tem:
                    for D in tem:
                        num += 1
                        yield f"data: {num}{D}\n\n"
                        time.sleep(1)
            yield f"data:{'Done'}\n\n"
        return Response(generator(notes_midi, Q), mimetype='text/event-stream')

        #def generate():
         #   for data in X:
          #      yield f"data: {data}\n\n"
           #     time.sleep(1)  # 每条数据后暂停1秒


    return app
