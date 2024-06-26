import librosa
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}.')

def Audio_seq_generator(scaler,song_ids,T_index1,T_index2,arousal,valence,n_mels):
    X=[]
    Y=torch.empty(0,59,2)
    for song_index in range(index1,index2):
        index = int(song_ids[song_index])
        audio,sr = librosa.load(f"{index}.mp3",sr=44100)
        audio=audio[:sr*45]
        S = librosa.feature.melspectrogram(y=audio,sr=sr,n_mels=n_mels,hop_length=441)
        S = S[:,:4500-45]
        S_dB = librosa.power_to_db(S, ref=np.max)
        SS = S_dB.reshape(n_mels,4500-45)
        X.append(SS)
        Y_seq = torch.tensor(np.stack((arousal[song_index,:],valence[song_index,:]),axis=1))
        Y_seq = Y_seq.reshape(1,Y_seq.shape[0],Y_seq.shape[1])
        Y = torch.cat((Y,Y_seq),dim=0)
    X=torch.tensor(np.array(X))
    #Y=torch.tensor(Y)

    #A,B,C=X.shape
    #X_flat=X.reshape(-1,1)
    #scaler.fit(X_flat.cpu())
    #X_N=scaler.transform(X_flat.cpu())
    #X_N=X_N.reshape(A,B,C)
    #X=torch.tensor(X_N,dtype=torch.float32)
    X = [X[:,:,i*50+1000:i*50+1500] for i in range(59)]
    X = torch.tensor(np.stack(X,axis=1))
    return X,Y

arousal = pd.read_csv('arousal.csv').to_numpy()[:,1:60]
valence = pd.read_csv('valence.csv').to_numpy()[:,1:60]
song_ids = pd.read_csv('valence.csv').to_numpy()[:,0]

scaler = StandardScaler()
X_train,Y_train =Audio_seq_generator(scaler,song_ids,0,150,arousal,valence,13)
X_val, Y_val = Audio_seq_generator(scaler,song_ids,151,200,arousal,valence,13)
X_train=X_train.to(device)
Y_train=Y_train.to(device)
print(X_train.shape,Y_train.shape)


class DynamicPCALayer(nn.Module):
    def __init__(self, n_components):
        super(DynamicPCALayer, self).__init__()
        self.n_components = n_components
        self.pca = None

    def fit(self, X):
        # 通过PCA拟合数据
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X)

    def forward(self, X):
        # 获取输入数据的形状
        batch_size, seq_len, feature_dim = X.size()

        # 将输入数据调整为 (batch_size * seq_len, feature_dim)
        X_flat = X.view(-1, feature_dim)

        if self.pca is None:
            # 初次调用时进行拟合
            self.fit(X_flat.cpu().detach().numpy())

        # 使用PCA转换数据
        X_pca = self.pca.transform(X_flat.cpu().detach().numpy())

        # 将结果转换回Tensor，并恢复形状为 (batch_size, seq_len, n_components)
        X_pca = torch.from_numpy(X_pca).float().to(X.device)
        X_pca = X_pca.view(batch_size, seq_len, -1)

        return X_pca

class RCNN(nn.Module):
    def __init__(self):
        super(RCNN,self).__init__()
        self.conv1 = nn.Conv1d(in_channels=13,out_channels=64,kernel_size = 10 ,stride = 1,padding = 0)
        self.conv2 = nn.Conv1d(in_channels=64,out_channels=64,kernel_size = 3 ,stride = 1,padding = 0)
        self.conv3 = nn.Conv1d(in_channels=64,out_channels=128,kernel_size = 3 ,stride = 1,padding = 0)
        self.conv4 = nn.Conv1d(in_channels=128,out_channels=128,kernel_size = 2 ,stride = 1,padding = 0)

        self.maxpooling4 = nn.AvgPool1d(kernel_size=4,stride=4,padding=0)
        self.maxpooling3 = nn.AvgPool1d(kernel_size=3,stride=3,padding=0)
        self.pca = DynamicPCALayer(8)
        self.flatten = nn.Flatten(start_dim=1)
        self.lstm = nn.LSTM(input_size=8,hidden_size=64,num_layers=5,batch_first=True)
        self.fc1 = nn.Linear(64,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,2)
        #self.fc3 = nn.Linear(32,2)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
    def forward(self,x):
        x1=[]
        for i in range(x.shape[0]):
            xx=x[i,:,:,:]
            xx=self.conv1(xx)
            xx=self.maxpooling4(xx)
            xx=self.conv2(xx)
            xx=self.maxpooling4(xx)
            xx=self.conv3(xx)
            xx=self.maxpooling4(xx)
            xx=self.conv4(xx)
            xx=self.maxpooling3(xx)
            xx=self.flatten(xx)
            #xx=self.conv5(xx)
            x1.append(xx)
        x1=torch.stack(x1,dim=0).float()

        x1 = self.pca(x1)
        #print(x1.shape)
        #torch.Size([1, 59, 256])
        #x1=torch.relu(x1.squeeze(3))
        #print(x1.shape)
        #torch.Size([5, 60, 1500, 64])
        #(batch_size, seq_len, input_size)
        #x2=[]
        #for i in range(x1.shape[0]):
        #    xxx=x[i,:,:,:]
        #    xxx,(hn,cn) = self.lstm(xxx)
        #    xxx=xxx[:,-1,:]
        #    x2.append(xxx)
        #x=torch.stack(x2,dim=0).float()
        x,(hn,cn) = self.lstm(x1)
        x = self.fc1(x)
        #x = self.relu(x)
        x = self.fc2(x)
        #x = self.relu(x)
        x = self.fc3(x)

        #x = self.relu(x)
        #x = self.tanh(x)
        return x

  def evaluation(model, X, Y):
    T=0
    S=Y.shape[0]*Y.shape[1]
    for i in range(X.shape[0]):
        y_hat=model(X[i:i+1,:,:,:])<0
        yy = Y[i:i+1,:,:]<0

        y_hat0=(y_hat[0,:,0])
        yy0=yy[0,:,0]

        y_hat1=y_hat[0,:,1]
        yy1=yy[0,:,1]
        T += ((yy0 == y_hat0) & (yy1==y_hat1)).int().sum().item()

    return T/S*100

def init_weights(m):
  if type(m)== torch.nn.Linear or type(m)== torch.nn.Conv1d :
    torch.nn.init.xavier_uniform_(m.weight)

# 模型初始化
model = RCNN().to(device)
model.apply(init_weights)
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(),lr=0.01)

epochs = 150
batch_size = 10

epoch_losses=[]
epoch_T_rate=[]
for epoch in range(epochs):
  model.train()
  total_loss = 0
  for i in range(int(X.shape[0]//batch_size)):
    start_index = i * batch_size
    end_index = start_index + batch_size
    inputs = X_train[start_index:end_index,:, :].float()#.to(device).float()
    targets = Y_train[start_index:end_index,:].float()#.to(device).float()
    # 前向传播
    outputs = model(inputs)
    loss = criterion(outputs, targets)

      # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    loss1= total_loss/int(X.shape[0]//batch_size)
  with torch.no_grad():
    model.eval()
    T_rate = evaluation(model,X_val,Y_val)


  print(f'Epoch {epoch+1}/{epochs}, Loss: {loss1:.4f}, T_rate: {T_rate:.2f}')
  epoch_losses.append(loss1)
  epoch_T_rate.append(T_rate)

plt.plot(epoch_losses)
plt.xlabel('Epoch')
plt.ylabel('Average cross entropy loss')
plt.show()
plt.plot(epoch_T_rate)
plt.xlabel('Epoch')
plt.ylabel('accuracy rate')
plt.show()

YY=(Y)#.int()
for a in range(150):
    YH=(model(X[a:a+1,:,:,:]))#.int()
    plt.scatter(YY[a,:,0].cpu(),YY[a,:,1].cpu(),c='b')
    plt.scatter(YH.cpu().detach().numpy()[0,:,0],YH.cpu().detach().numpy()[0,:,1],c='g')
plt.show()
