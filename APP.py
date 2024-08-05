import akshare as ak
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm
from datetime import datetime, timedelta

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
stock_zh_spot_df = ak.stock_zh_a_spot_em()  # 获取实时数据
stock_zh_spot_data = stock_zh_spot_df[stock_zh_spot_df['名称'] != '']
stock_zh_spot_data = stock_zh_spot_data[~stock_zh_spot_data['名称'].str.contains('[a-zA-Z]')]
stock_zh_spot_data = stock_zh_spot_data[~stock_zh_spot_data['代码'].str.startswith('8')]
stock_zh_spot_data = stock_zh_spot_data[~stock_zh_spot_data['代码'].str.startswith('68')]
stock_zh_spot_data = stock_zh_spot_data[~stock_zh_spot_data['代码'].str.startswith('4')]
stock_zh_spot_data = stock_zh_spot_data.dropna()
stock_zh_spot_data['代码'] = stock_zh_spot_data['代码']
codes_names = np.array(stock_zh_spot_data['代码'])
now = datetime.now()
formatted_time = now.strftime('%Y%m%d')
past_date = now - timedelta(days=120)
formatted_past_date = past_date.strftime('%Y%m%d')
length = len(codes_names)
df = pd.DataFrame([])
for i in tqdm(range(length), desc="Processing items"):
    try:
        data_df = ak.stock_zh_a_hist(symbol=codes_names[i], period="daily", start_date=formatted_past_date, end_date=formatted_time, adjust="hfq")  # 日度数据，后复权
        data_df['stock_id'] = codes_names[i]
        df = df.append(data_df)
    except Exception as e:
        print(f"An error occurred: {e}")
filtered_df = (
    df.groupby('stock_id')
    .filter(lambda x: (x['换手率'] >= 0.3).all() and not x.isnull().values.any())
)
df = filtered_df.reset_index(drop=True)
df['成交额'] = df['成交额'] / 10000000
grouped = df.groupby('stock_id')
samples = []
ID = []
for _, group in tqdm(grouped):
    product_samples = group.values
    id =product_samples[-1][11]
    DATA = product_samples[ :, [1, 2, 3, 4, 5, -2]][-60:, :]
    if DATA.shape[0]==60:
        samples.append(DATA) #开、收、高、低、成交额、换手率
        ID.append(id)
train_data = np.array(samples).astype(np.float32)
Trian_data = torch.tensor(train_data)
print(fr'数据处理完毕:size{Trian_data.shape}')
batch_size = 500
dataset_test = TensorDataset(Trian_data)
test_order = DataLoader(dataset_test, shuffle=False, batch_size=batch_size)

P =[]
for k in range(5):
    model_path1 = fr'./weights/model_APP_{k}.pt'
    class model(nn.Module):
        def __init__(self,
                     fc1_size=2000,
                     fc2_size=1000,
                     fc3_size=100,
                     fc1_dropout=0.2,
                     fc2_dropout=0.2,
                     fc3_dropout=0.2,
                     num_of_classes=50):
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
                nn.Linear(fc3_size, 1),

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
                                bidirectional=True)
            hidden_dim = 1
            self.l = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=4, batch_first=True,
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
    model.load_state_dict(torch.load(model_path1, map_location=device))
    model.to(device)
    model.eval()
    PREDICT = []
    with torch.no_grad():
        test_order = tqdm(test_order)
        for SEQ in test_order:
            SEQ = SEQ[0]
            output = model(SEQ.to(device))
            PREDICT.extend(output.cpu().numpy())
    P.append(PREDICT)
PREDICT = np.mean(P,axis=0)
ndarray_flat = PREDICT.flatten()
df = pd.DataFrame({
    '模型打分': ndarray_flat,
    'stock_id': ID
})
df.to_csv('output.csv', index=False)


