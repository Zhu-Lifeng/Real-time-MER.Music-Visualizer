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
for element in midi_stream.flatten():
  if isinstance(element, tempo.MetronomeMark):
    MM = element.number
  if isinstance(element,note.Rest):
    notes_midi.append([[],element.duration.quarterLength])
  if isinstance(element, note.Note):
    notes_midi.append([[element.pitch.nameWithOctave],element.duration.quarterLength])
  if isinstance(element,chord.Chord):
    chords=[]
    for N in element.notes:
      chords.append(N.pitch.nameWithOctave)
    notes_midi.append([chords,element.duration.quarterLength])

print(MM)
for pitch, duration in notes_midi[0:10]:
    print(pitch,duration)


# Audio: Input processing
audio_file='/content/drive/MyDrive/MSC Project/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.wav'
y, sr = librosa.load(audio_file,sr=44100)
pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr,n_fft=4410,threshold=0.1)
notes_audio=[]
for i in range(magnitudes.shape[1]):
  column = magnitudes[:, i]
  sorted_indices = np.argsort(column)[::-1]
  top_indices = sorted_indices[:20]
  top_freqs = pitches[top_indices, i]
  top_mags = column[top_indices]
  notes_audio.append(np.column_stack((np.array(top_freqs),np.array(top_mags))))
