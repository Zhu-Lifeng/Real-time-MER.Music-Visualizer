import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import joblib

class DynamicPCALayer(nn.Module):
    def __init__(self, n_components):
        super(DynamicPCALayer, self).__init__()
        self.n_components = n_components
        self.pca = None

    def fit(self, X):
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X)

    def forward(self, X):
        A, batch_size, seq_len, feature_dim = X.size()
        X_flat = X.view(-1, feature_dim)
        if self.pca is None:
            self.fit(X_flat.cpu().detach().numpy())
        X_pca = self.pca.transform(X_flat.cpu().detach().numpy())
        X_pca = torch.from_numpy(X_pca).float().to(X.device)
        X_pca = X_pca.view(A, batch_size, seq_len, -1)
        return X_pca


class RCNN(nn.Module):
    def __init__(self):
        super(RCNN, self).__init__()
        self.conv11 = nn.Conv1d(in_channels=24, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.conv21 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=4, stride=1, padding=0)
        self.conv31 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.conv41 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)

        self.conv12 = nn.Conv1d(in_channels=1, out_channels=12, kernel_size=5, stride=1, padding=0)
        self.conv22 = nn.Conv1d(in_channels=12, out_channels=12, kernel_size=4, stride=1, padding=0)
        self.conv32 = nn.Conv1d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=0)
        self.conv42 = nn.Conv1d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=0)

        self.conv13 = nn.Conv1d(in_channels=24, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.conv23 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=4, stride=1, padding=0)
        self.conv33 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.conv43 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)

        self.maxpooling2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.flatten = nn.Flatten(start_dim=1)

        self.pca1 = DynamicPCALayer(8)
        self.pca2 = DynamicPCALayer(4)
        self.pca3 = DynamicPCALayer(8)

        # self.lstm = nn.LSTM(input_size=64,hidden_size=128,num_layers=5,batch_first=True)
        self.GRU = nn.GRU(input_size=20, hidden_size=128, num_layers=5, batch_first=True)
        self.fc11 = nn.Linear(128, 256)
        self.fc21 = nn.Linear(256, 1)
        self.fc12 = nn.Linear(128, 256)
        self.fc22 = nn.Linear(256, 1)
        # self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.drop = torch.nn.Dropout(0.1)

    def forward(self, x):
        x1 = x[:, :, :, 0:24, :]
        x2 = x[:, :, :, 24:25, :]
        x3 = x[:, :, :, 25:49, :]
        X11 = []
        for i in range(x1.shape[0]):
            X12 = []
            for j in range(x1.shape[1]):
                xx = x1[i, j, :, :, :]
                xx = self.conv11(xx)
                xx = self.maxpooling2(xx)
                xx = self.conv21(xx)
                xx = self.maxpooling2(xx)
                xx = self.conv31(xx)
                xx = self.maxpooling2(xx)
                xx = self.conv41(xx)
                xx = self.maxpooling2(xx)
                xx = self.flatten(xx)
                X12.append(xx)
            X12 = torch.stack(X12, dim=0).float()
            X11.append(X12)
        X11 = torch.stack(X11, dim=0).float()
        X11 = self.pca1(X11)

        X21 = []
        for i in range(x2.shape[0]):
            X22 = []
            for j in range(x2.shape[1]):
                xx = x2[i, j, :, :, :]
                xx = self.conv12(xx)
                xx = self.maxpooling2(xx)
                xx = self.conv22(xx)
                xx = self.maxpooling2(xx)
                xx = self.conv32(xx)
                xx = self.maxpooling2(xx)
                xx = self.conv42(xx)
                xx = self.maxpooling2(xx)
                xx = self.flatten(xx)
                X22.append(xx)
            X22 = torch.stack(X22, dim=0).float()
            X21.append(X22)
        X21 = torch.stack(X21, dim=0).float()
        X21 = self.pca2(X21)

        X31 = []
        for i in range(x3.shape[0]):
            X32 = []
            for j in range(x3.shape[1]):
                xx = x3[i, j, :, :, :]
                xx = self.conv13(xx)
                xx = self.maxpooling2(xx)
                xx = self.conv23(xx)
                xx = self.maxpooling2(xx)
                xx = self.conv33(xx)
                xx = self.maxpooling2(xx)
                xx = self.conv43(xx)
                xx = self.maxpooling2(xx)
                xx = self.flatten(xx)
                X32.append(xx)
            X32 = torch.stack(X32, dim=0).float()
            X31.append(X32)
        X31 = torch.stack(X31, dim=0).float()
        X31 = self.pca3(X31)

        X = torch.cat([X11, X21, X31], dim=3)

        XX3 = []
        for i in range(x1.shape[0]):
            x, _ = self.GRU(X[i, :, :, :])
            XX3.append(x[:, -1, :])

        x = torch.stack(XX3, dim=0).float()

        x1 = self.drop(x)
        x1 = self.fc11(x1)
        x1 = self.relu(x1)
        x1 = self.fc21(x1)

        x2 = self.fc12(x)
        x2 = self.relu(x2)
        x2 = self.fc22(x2)

        x = torch.cat([x1, x2], dim=2)
        # x = self.tanh(x)
        return x

    def save_model(self, model_path, pca_paths):
        torch.save(self.state_dict(), model_path)
        joblib.dump(self.pca1.pca, pca_paths[0])
        joblib.dump(self.pca2.pca, pca_paths[1])
        joblib.dump(self.pca3.pca, pca_paths[2])

    def load_model(self, model_path, pca_paths):
        self.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
        self.pca1.pca = joblib.load(pca_paths[0])
        self.pca2.pca = joblib.load(pca_paths[1])
        self.pca3.pca = joblib.load(pca_paths[2])