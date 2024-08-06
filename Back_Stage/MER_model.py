import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import joblib


class DynamicPCALayer_Seq(nn.Module):
    def __init__(self, n_components):
        super(DynamicPCALayer_Seq, self).__init__()
        self.n_components = n_components
        self.pca = None

    def fit(self, X):
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X)

    def forward(self, X):
        batch_size, seq_len, feature_dim = X.size()
        # batch_size, feature_dim = X.size()
        X_flat = X.view(-1, feature_dim)
        if self.pca is None:
            self.fit(X_flat.cpu().detach().numpy())

        X_pca = self.pca.transform(X_flat.cpu().detach().numpy())
        X_pca = torch.from_numpy(X_pca).float().to(X.device)
        X_pca = X_pca.view(batch_size, seq_len, -1)
        # X_pca = X_pca.view(batch_size, -1)
        return X_pca


class RCNN(nn.Module):
    def __init__(self):
        super(RCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=53, out_channels=48, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv1d(in_channels=48, out_channels=64, kernel_size=4, stride=1, padding=0)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=0)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=2, stride=1, padding=0)

        self.maxpooling4 = nn.MaxPool1d(kernel_size=4, stride=4, padding=0)
        self.maxpooling3 = nn.MaxPool1d(kernel_size=3, stride=3, padding=0)
        self.maxpooling2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.flatten = nn.Flatten(start_dim=1)

        self.pca = DynamicPCALayer_Seq(72)
        # self.GRU = nn.GRU(input_size=64, hidden_size=128, num_layers=5, batch_first=True)
        self.LSTM = nn.LSTM(input_size=72, hidden_size=256, num_layers=10, batch_first=True)

        # self.pca2 = DynamicPCALayer_Non_Seq(32)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        # self.fc3 = nn.Linear(256, 256)
        # self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 2)

        self.relu = nn.ReLU()
        self.drop = torch.nn.Dropout(0.5)

    def forward(self, x, h=None):
        X11 = []
        for i in range(x.shape[0]):
            xx = x[i, :, :, :]
            xx = self.conv1(xx)
            xx = self.maxpooling4(xx)
            xx = self.conv2(xx)
            xx = self.maxpooling3(xx)
            xx = self.conv3(xx)
            xx = self.maxpooling2(xx)
            xx = self.conv4(xx)
            xx = self.maxpooling2(xx)

            xx = self.flatten(xx)
            X11.append(xx)
        X = torch.stack(X11, dim=0).float()
        X = self.pca(X)
        # x,_ = self.GRU(X)
        x, _ = self.LSTM(X)
        x = x[:, -1, :]
        # x=self.pca2(x)

        # x = torch.cat((x,x_f),dim=2)

        x = self.drop(x)
        x1 = self.fc1(x)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)
        # x1 = self.relu(x1)
        # x1 = self.fc3(x1)
        # x1 = self.relu(x1)
        # x1 = self.fc4(x1)
        x1 = self.relu(x1)
        x = self.fc5(x1)
        return x

    def save_model(self, model_path, pca_paths):
        torch.save(self.state_dict(), model_path)
        joblib.dump(self.pca.pca, pca_paths[0])
        # joblib.dump(self.pca2.pca, pca_paths[1])
        # joblib.dump(self.pca3.pca, pca_paths[2])

    def load_model(self, model_path, pca_paths):
        self.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
        self.pca.pca = joblib.load(pca_paths[0])
        # self.pca2.pca = joblib.load(pca_paths[1])
        # self.pca3.pca = joblib.load(pca_paths[2])
