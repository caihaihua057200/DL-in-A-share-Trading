
import numpy as np
import pandas as pd
import torch
from sklearn import metrics
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Normalize
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim import lr_scheduler
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import time
from tqdm import tqdm
import os
if not os.path.exists('./weights'):
    os.makedirs('./weights')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df = pd.read_csv('./data/20240801.csv')

df['成交额'] = df['成交额'] / 10000000
df = df.groupby('stock_id').filter(lambda x: len(x) >= 180)  # 少于180交易日的股票不要
df = df.groupby('stock_id').apply(lambda x: x.iloc[20:])  # 刚上市的20个交易日不要
df.reset_index(drop=True, inplace=True)


start_time = time.time()
grouped = df.groupby('stock_id')
samples = []
label = []
for _, group in tqdm(grouped):
    product_samples = group.values
    num_samples = len(product_samples)
    if num_samples < 180:
        continue
    for i in range(num_samples - 85):
        LLL = product_samples[i:i + 86, 2:6].astype(np.float32)
        LLLL = product_samples[i:i + 86, 11:12].astype(np.float32)
        LLLLL = product_samples[i:i + 86, 9:10].astype(np.float32)
        if (np.any(np.isnan(LLL)) or np.any(LLL <= 0) or np.any(np.isnan(LLLL)) or
                np.any(LLLL < 0.1) or np.any(LLLL > 40) or
                np.any(LLLLL > 20) or np.any(LLLLL < -10)):
            pass
        else:
            sample = product_samples[i:i + 60, 2:-1]

            l = product_samples[i + 60:i + 61, 2:3]
            Hl = product_samples[i + 60:i + 61, 4:5]
            if l != Hl:
                ll = np.mean(product_samples[i + 61:i + 86, 5:6])
                lll = ((ll - l) / l)*100
                label.append(lll)
                samples.append(sample[ :, [0, 1, 2, 3, 4, -1]]) #开、收、高、低、成交额、换手率

train_data = np.array(samples)
train_data = train_data.astype(np.float32)
train_label = np.array(label).astype(np.float32)
print('训练集：',train_data.shape)
print(train_data.shape)
end_time = time.time()
execution_time = end_time - start_time
print(f"代码执行时间为: {execution_time} 秒")

MODEL = r'APP'
batch_size = 40000
learning_rate = 0.001
N = train_data.shape[0]
num_epochs = 200
k = 3
cl_splits = KFold(n_splits=k, shuffle=True, random_state=666)
softmax_function = nn.Softmax(dim=1)

L5 = []
L5_v = []
T_AUC = []

for fold, (train_idx, test_idx) in enumerate(cl_splits.split(np.arange(N))):
        Train_data = train_data[train_idx]
        Train_label = train_label[train_idx]

        Test_data = train_data[test_idx]
        Test_label = train_label[test_idx]

        Train_data = torch.tensor(Train_data, dtype=torch.float32)
        Train_label = torch.tensor(Train_label, dtype=torch.float32)

        Test_data = torch.tensor(Test_data, dtype=torch.float32)
        Test_label = torch.tensor(Test_label, dtype=torch.float32)

        dataset_train = TensorDataset(Train_data, Train_label)
        dataset_val = TensorDataset(Test_data, Test_label)
        print('Fold {}'.format(fold + 1))
        train_loader = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)
        val_loader = DataLoader(dataset_val, shuffle=True, batch_size=batch_size)

        class model(nn.Module):
            def __init__(self,
                         fc1_size=2000,
                         fc2_size=1000,
                         fc3_size=100,
                         fc1_dropout=0.2,
                         fc2_dropout=0.2,
                         fc3_dropout=0.2,
                         num_of_classes=1):
                super(model, self).__init__()

                self.f_model = nn.Sequential(
                    nn.Linear(3296, fc1_size),  # 887
                    nn.BatchNorm1d(fc1_size),
                    nn.ReLU(),
                    nn.Dropout(fc1_dropout),
                    nn.Linear(fc1_size, fc2_size),
                    nn.BatchNorm1d(fc2_size),
                    nn.ReLU(),
                    nn.Dropout(fc2_dropout),
                    nn.Linear(fc2_size, fc3_size),
                    nn.BatchNorm1d(fc3_size),
                    nn.ReLU(),
                    nn.Dropout(fc3_dropout),
                    nn.Linear(fc3_size, num_of_classes),

                )

                self.conv_layers1 = nn.Sequential(
                    nn.Conv1d(6, 16, kernel_size=1),
                    nn.BatchNorm1d(16),
                    nn.Dropout(fc3_dropout),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2),
                    nn.Conv1d(16, 32, kernel_size=1),
                    nn.BatchNorm1d(32),
                    nn.Dropout(fc3_dropout),
                    nn.ReLU(),
                )

                self.conv_2D = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=2),
                    nn.BatchNorm2d(16),
                    nn.Dropout(fc3_dropout),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),
                    nn.Conv2d(16, 32, kernel_size=2),
                    nn.BatchNorm2d(32),
                    nn.Dropout(fc3_dropout),
                    nn.ReLU(),
                )
                hidden_dim = 32
                self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=4, batch_first=True,
                                    # dropout=fc3_dropout,
                                    bidirectional=True)
                hidden_dim = 1
                self.l = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=4, batch_first=True,
                                 # dropout=fc3_dropout,
                                 bidirectional=True)
                for name, module in self.named_modules():
                    if isinstance(module, nn.Linear):
                        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                    if isinstance(module, nn.Conv2d):
                        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                    if isinstance(module, nn.Conv1d):
                        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')

            def forward(self, x):
                apply = torch.narrow(x, dim=-1, start=0, length=1)[:, -90:, ].squeeze(1)
                redeem = torch.narrow(x, dim=-1, start=1, length=1)[:, -90:, ].squeeze(1)
                apply, _ = self.l(apply)
                redeem, _ = self.l(redeem)
                apply = torch.reshape(apply, (apply.shape[0], apply.shape[1] * apply.shape[2]))
                redeem = torch.reshape(redeem, (redeem.shape[0], redeem.shape[1] * redeem.shape[2]))

                ZFF = torch.narrow(x, dim=-1, start=2, length=1)[:, -90:, ].squeeze(1)
                HS = torch.narrow(x, dim=-1, start=3, length=1)[:, -90:, ].squeeze(1)
                ZFF, _ = self.l(ZFF)
                HS, _ = self.l(HS)
                ZFF = torch.reshape(ZFF, (ZFF.shape[0], ZFF.shape[1] * ZFF.shape[2]))
                HS = torch.reshape(HS, (HS.shape[0], HS.shape[1] * HS.shape[2]))

                min_vals, _ = torch.min(x, dim=1, keepdim=True)
                max_vals, _ = torch.max(x, dim=1, keepdim=True)
                x = (x - min_vals) / (max_vals - min_vals + 0.00001)

                xx = x.unsqueeze(1)
                xx = self.conv_2D(xx)
                xx = torch.reshape(xx, (xx.shape[0], xx.shape[1] * xx.shape[2] * xx.shape[3]))
                x = x.transpose(1, 2)
                x = self.conv_layers1(x)
                out = x.transpose(1, 2)
                out2, _ = self.lstm(out)
                out2 = torch.reshape(out2, (out2.shape[0], out2.shape[1] * out2.shape[2]))

                IN = torch.cat((xx, out2, apply, redeem, ZFF, HS), dim=1)
                out = self.f_model(IN)
                return out


        model = model()
        model.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), weight_decay=1e-5)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
        L_train = []
        L_val = []
        AUC = []
        min_validation_loss = 0
        for epoch in range(num_epochs):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            train_running_loss = 0.0
            counter = 0
            model.train()
            for seq, y in tqdm(train_loader):
                counter += 1
                output = model(seq.to(device))
                loss = criterion(output, y.squeeze(1).to(device))
                train_running_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            TL = train_running_loss / counter
            L_train.append(TL)
            model.eval()
            PREDICT = []
            TRUE = []
            counter = 0
            with torch.no_grad():
                current_test_loss = 0.0
                for SEQ, Z in tqdm(val_loader):
                    counter += 1
                    output = model(SEQ.to(device))
                    loss = criterion(output, Z.squeeze(1).to(device))
                    current_test_loss += loss.item()
                    PREDICT.extend(output.cpu().numpy())
                    TRUE.extend(Z.cpu().numpy())
                T_loss = current_test_loss / counter
                L_val.append(T_loss)
                PP = np.array(PREDICT)
                TT = np.array(TRUE)
                flattened_array1 = PP.flatten()
                flattened_array2 = TT.flatten()
                correlation_matrix = np.corrcoef(flattened_array1, flattened_array2)
                corr = correlation_matrix[0, 1]
                print("Train loss: ", TL, "Val loss: ", T_loss, 'correlation_value', corr)

                ################################################################################################################
                if min_validation_loss < corr:
                    min_validation_loss = corr
                    best_epoch = epoch
                    print('Max pr_auc ' + str(min_validation_loss) + ' in epoch ' + str(best_epoch))
                    torch.save(model.state_dict(), fr"./weights/model_{MODEL}_{fold}.pt")
                AUC.append(corr)

        L5.append(L_train)
        L5_v.append(L_val)
        T_AUC.append(AUC)

fig1, ax1 = plt.subplots()
for i, fold_train_loss in enumerate(L5):
    ax1.plot(fold_train_loss, label=f"Fold {i + 1}")
ax1.legend(loc='upper right')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss')
ax1.set_title('5-fold cross-validation training losses')

# 绘制验证损失函数折线图
fig2, ax2 = plt.subplots()
for i, fold_val_loss in enumerate(L5_v):
    ax2.plot(fold_val_loss, label=f"Fold {i + 1}")
ax2.legend(loc='upper right')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Validation Loss')
ax2.set_title('5-fold cross-validation validation losses')

fig3, ax3 = plt.subplots()
for i, fold_val_loss in enumerate(T_AUC):
    ax3.plot(fold_val_loss, label=f"Fold {i + 1}")
ax3.legend(loc='upper right')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Test-set corr')
ax3.set_title('Test-set corr')

fig1.savefig(f"./{MODEL}Training Loss.png")
fig2.savefig(f"./{MODEL}Validation Loss.png")
fig3.savefig(f"./{MODEL}corr.png")
