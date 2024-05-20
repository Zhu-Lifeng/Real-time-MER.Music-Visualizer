# MIDI Part => the Output
#pip install mido music21
from music21 import converter, instrument, midi, note, chord
import matplotlib.pyplot as plt

midi_file = '/content/drive/MyDrive/MSC Project/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi'
midi_stream = converter.parse(midi_file)

for element in midi_stream.flatten().notes[0:20]:
  if isinstance(element, chord.Chord):
    print(element, element.notes)

for element in midi_stream.flatten().notes[0:20]:
  if isinstance(element, note.Note):
    print(element, element.pitch, element.duration.quarterLength)


# Audio Part => the Input

!pip install librosa
!pip install numpy
!pip install matplotlib

import librosa
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# to_stream, waiting to be learned
stream = librosa.stream(file_path,block_length=256,frame_length=4096,hop_length=1024)

# encoding
y,sr=librosa.load('file_path')
# y: audio time series (np.ndarray[shape=(n,) or (...,n)(multi-channel)])
# sr: sampling rate (number >0)
pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, threshold=0.1 )
# return the active pitches and their corresponse magnitudes
notes=[]
a = 1000
B=[pitches[:,a],magnitudes[:,a]]
for i in range(pitches.shape[0]):
  if pitches[i,a] != 0:
    notes.append([pitches[i,a],magnitudes[i,a]] )

print(notes)
