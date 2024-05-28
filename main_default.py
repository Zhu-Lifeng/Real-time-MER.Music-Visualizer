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
pitch_88=[27.50,29.14,30.87,32.70,34.65,36.71,38.89,41.20,43.65,46.25,49.00,51.91,
         55.00,58.27,61.74,65.41,69.30,73.42,77.78,82.41,87.31,92.50,
         98.00,103.83,110.00,116.54,123.47,130.81,138.59,146.83,155.56,
         164.81,174.61,185.00,196.00,207.65,220.00,233.08,246.94,261.63,
         277.18,293.66,311.13,329.23,349.23,369.99,392.00,415.30,440.00,
         466.16,493.88,523.25,554.37,587.33,622.25,659.26,698.46,739.99,
         783.99,830.61,880.00,932.33,987.77,1046.50,1108.73,1174.66,1244.51,
         1318.51,1396.91,1479.98,1567.98,1661.22,1760.00,1864.66,1975.53,
         2093.00,2217.46,2349.32,2489.02,2637.02,2793.83,2959.96,3135.96,
         3322.44,3520.00,3729.31,3951.07,4186.01]
PITCH_88=['A0','A#0','B0',
          'C1','C#1','D1','D#1','E1','F1','F#1','G1','G#1','A1','A#1','B1',
          'C2','C#2','D2','D#2','E2','F2','F#2','G2','G#2','A2','A#2','B2',
          'C3','C#3','D3','D#3','E3','F3','F#3','G3','G#3','A3','A#3','B3',
          'C4','C#4','D4','D#4','E4','F4','F#4','G4','G#4','A4','A#4','B4',
          'C5','C#5','D5','D#5','E5','F5','F#5','G5','G#5','A5','A#5','B5',
          'C6','C#6','D6','D#6','E6','F6','F#6','G6','G#6','A6','A#6','B6',
          'C7','C#7','D7','D#7','E7','F7','F#7','G7','G#7','A7','A#7','B7',
          'C8']


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

def pitch_check(x):
    index = (np.abs(np.array(pitch_88) - x)).argmin()
    return index

notes_audio=np.zeros([pitches.shape[1],88])
PM=np.stack((pitches,magnitudes))
print(PM.shape,notes_audio.shape)

for i in range(magnitudes.shape[1]):
  for j in range(magnitudes.shape[0]):
    if magnitudes[j,i] > 0:
      index = pitch_check(pitches[j,i])
      notes_audio[i,index] = magnitudes[j,i]

print(notes_audio[5000,:])


# Model
import torch
class MRNN(torch.nn.Module):
  def __init__(self,hidden_size):
    super(MRNN, self).__init__()
    self.seq_length = 88
    self.WI = torch.nn.Parameter(torch.randn(self.seq_length,hidden_size)*0.01).float()
    self.WR = torch.nn.Parameter(torch.randn(hidden_size,hidden_size)*0.01).float()
    self.b = torch.nn.Parameter(torch.zeros(hidden_size)).float()
    self.linear = torch.nn.Linear(hidden_size, 88*3)
  def initial_state(self, batch_size):
    return torch.zeros(1, batch_size, hidden_size).float()

  def forward(self, X, state=None):
    if state is None:
      state = self.initial_state(X.shape[1]).float()
    outputs=[]
    for element in X:
      #print('element:',element.shape)
      state = torch.tanh(torch.matmul(element.float(),self.WI)+torch.matmul(state.float(),self.WR)+self.b)
      outputs.append(state.float())
      #print('state:',state.shape)
    outputs = torch.stack(outputs)
    #print(outputs.shape)
    #print(state.shape)
    #state = state.reshape(1, state.shape[0], state.shape[1])
    outputs = self.linear(outputs)
    return state, outputs

def seq_generater(X,Y,batch_size,num_steps):
  len_macro_subseq = (len(X)-1)//batch_size
  num_elements = len_macro_subseq*batch_size
  Xs = torch.tensor(X[:num_elements,:])
  Ys = torch.tensor(Y[:num_elements,:])
  num_subseqs = Xs.shape[0] // num_steps
  for i in range(0,num_subseqs * num_steps, num_steps):
    XX=Xs[i:i+num_steps , :].float()
    YY=Ys[i:i+num_steps , :].float()
    yield XX,YY
def init_weights(m):
  if type(m)== torch.nn.Linear or type(m) == torch.nn.Conv2d:
    torch.nn.init.xavier_uniform_(m.weight)

hidden_size=400
lr=0.001
num_epochs=20
batch_size=256
num_steps=500

ScT=MRNN(hidden_size).to(device)
ScT.apply(init_weights)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(ScT.parameters(), lr=lr, betas=(0.8, 0.8))
epoch_losses = []
for epoch in range(num_epochs):
  print(f'\nEpoch {epoch + 1}/{num_epochs}.')
  Train_iter=seq_generater(notes_audio,notes_midi,batch_size,num_steps)
  state = ScT.initial_state(batch_size).to(device)
  losses=[]
  for X,Y in Train_iter:
    X, Y = X.to(device), Y.to(device)
    state, outputs = ScT(X,state)
    Y_flat = Y.reshape(-1).long()
    outputs_flat=outputs.reshape(-1,outputs.shape[2])
    l = loss(outputs_flat,Y_flat)
    optimizer.zero_grad()
    l.backward()
    optimizer.step()
    losses.append(float(l))
    state = state.clone().detach()
  epoch_losses.append(np.mean(losses))
  print(f'Epoch {epoch + 1}/{num_epochs}. Loss: {epoch_losses[-1]:.5f}.')

plt.plot(epoch_losses)
plt.xlabel('Epoch')
plt.ylabel('Average cross entropy loss')
plt.show()
