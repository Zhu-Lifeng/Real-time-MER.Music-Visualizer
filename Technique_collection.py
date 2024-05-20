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

