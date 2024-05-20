#!pip install mido music21
#!pip install librosa
#!pip install numpy
#!pip install matplotlib

from music21 import converter, instrument, midi, note, chord
import matplotlib.pyplot as plt
import librosa
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

midi_file = '/content/drive/MyDrive/MSC Project/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi'
midi_stream = converter.parse(midi_file)

audio_file='/content/drive/MyDrive/MSC Project/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.wav'
y, sr = librosa.load(audio_file)

notes=[]
for element in midi_stream.flatten().notes:
  if isinstance(element, note.Note):
    notes.append([[element.pitch.nameWithOctave],element.duration.quarterLength])
  if isinstance(element,chord.Chord):
    chords=[]
    for N in element.notes:
      chords.append(N.pitch.nameWithOctave)
    notes.append([chords,element.duration.quarterLength])

for pitch, duration in notes[0:10]:
    print(pitch,duration)
