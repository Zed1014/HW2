from scipy.io import loadmat
import torch
import numpy as np
from sklearn.model_selection import KFold
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from audtorch.metrics.functional import pearsonr
from sklearn.metrics import r2_score
import scipy.io as io

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.fc(output[:, -1, :])
        return output


def fit(model, X_train, y_train, batch_size, num_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    dataset = TensorDataset(X_train.T, y_train.T)
    # 创建一个数据加载器，自动分批次加载数据
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    for epoch in range(num_epochs):
        sum_loss = 0
        for X, y in dataloader:  ###分批训练
            optimizer.zero_grad()
            y_pred = model(X.unsqueeze(2).permute(0,2,1))
            loss = criterion(y_pred, y)
            loss.backward()
            sum_loss = sum_loss + loss.item()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {sum_loss:.4f}')
    return model

data = loadmat('bined_spk.mat')
bined_spk = torch.tensor(data['bined_spk'],dtype=torch.float).to('cuda')
data = loadmat('trial_pos.mat')
trial_pos = torch.tensor(data['trial_pos'].T,dtype=torch.float).to('cuda')
data = loadmat('trial_velocity.mat')
trial_velocity = torch.tensor(data['trial_velocity'].T,dtype=torch.float).to('cuda')
data = loadmat('trial_acceleration.mat')
trial_acceleration = torch.tensor(data['trial_acceleration'].T,dtype=torch.float).to('cuda')

CC = torch.zeros((10,3),dtype=float)
MSE = torch.zeros((10,3),dtype=float)
R2 = np.zeros((10,3),dtype=float)
prediction =torch.zeros((10,1),dtype=float)

batch_size = 32
num_epochs =60
model = LSTM(147, 512, 3).to(device)
criterion = nn.MSELoss().to(device)


# 设置交叉验证的折叠数
n_splits = 10
# 创建 KFold 对象并混洗数据
kf = KFold(n_splits=n_splits, shuffle=True, random_state=1234)
count = 0
for train_index, test_index in kf.split(bined_spk.T):
    X_train, X_test = bined_spk[:, train_index], bined_spk[:, test_index]
    y_train, y_test = trial_velocity[:, train_index], trial_velocity[:, test_index]

    # 训练模型
    net = fit(model, X_train, y_train, batch_size, num_epochs)

    # 测试模型
    with torch.no_grad():
        y_pred = net(X_test.unsqueeze(1).permute(2,1,0)).T

    CC[count,:] = torch.concat((pearsonr(y_test[0,:],y_pred[0,:]),pearsonr(y_test[1,:],y_pred[1,:]),pearsonr(y_test[2,:],y_pred[2,:])))
    MSE[count,:] = torch.concat((torch.tensor([criterion(y_test[0,:] , y_pred[0,:])]),torch.tensor([criterion(y_test[1,:] , y_pred[1,:])]),torch.tensor([criterion(y_test[2,:] , y_pred[2,:])])))
    R2[count,:] = np.concatenate((np.array([r2_score(y_test[0,:].cpu().numpy(), y_pred[0,:].cpu().numpy())]),
                                  np.array([r2_score(y_test[1,:].cpu().numpy(), y_pred[1,:].cpu().numpy())]),
                                  np.array([r2_score(y_test[2,:].cpu().numpy(), y_pred[2,:].cpu().numpy())])))

    print("CC:", CC[count,:].mean().item())
    print("MSE:", MSE[count, :].mean().item())
    print("R2:", R2[count, :].mean().item())
    count = count + 1
    mat_path = 'LSTM/velocity/CC.mat'
    io.savemat(mat_path, {'CC': CC.detach().numpy()})
    mat_path = 'LSTM/velocity/MSE.mat'
    io.savemat(mat_path, {'MSE': MSE.detach().numpy()})
    mat_path = 'LSTM/velocity/R2.mat'
    io.savemat(mat_path, {'R2': R2})
