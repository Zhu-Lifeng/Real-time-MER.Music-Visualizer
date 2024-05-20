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

# MIDI: Output processing
midi_file = '/content/drive/MyDrive/MSC Project/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi'
midi_stream = converter.parse(midi_file)

notes_midi=[]
for element in midi_stream.flatten().notes:
  if isinstance(element, note.Note):
    notes_midi.append([[element.pitch.nameWithOctave],element.duration.quarterLength])
  if isinstance(element,chord.Chord):
    chords=[]
    for N in element.notes:
      chords.append(N.pitch.nameWithOctave)
    notes_midi.append([chords,element.duration.quarterLength])

for pitch, duration in notes_midi[0:10]:
    print(pitch,duration)


# Audio: Input processing
audio_file='/content/drive/MyDrive/MSC Project/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.wav'
y, sr = librosa.load(audio_file)

pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, threshold=0.01)
notes_audio=[]
for j in range(pitches.shape[1]):
  N=[]
  for i in range(pitches.shape[0]):
    if pitches[i,j] != 0:
      N.append([pitches[i,j],magnitudes[i,j]])
  notes_audio.append(N)
for X in notes_audio[10:50]:
    print(X)
