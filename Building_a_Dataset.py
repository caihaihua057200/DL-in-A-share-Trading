import akshare as ak
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# 检查是否存在 data 文件夹，如果不存在则创建
if not os.path.exists('./data'):
    os.makedirs('./data')

stock_zh_spot_df = ak.stock_zh_a_spot_em()  # 获取实时数据
stock_zh_spot_data = stock_zh_spot_df[stock_zh_spot_df['名称'] != '']
stock_zh_spot_data = stock_zh_spot_data[~stock_zh_spot_data['名称'].str.contains('[a-zA-Z]')]
stock_zh_spot_data = stock_zh_spot_data[~stock_zh_spot_data['代码'].str.startswith('8')]
stock_zh_spot_data = stock_zh_spot_data[~stock_zh_spot_data['代码'].str.startswith('68')]
stock_zh_spot_data = stock_zh_spot_data[~stock_zh_spot_data['代码'].str.startswith('4')]
stock_zh_spot_data = stock_zh_spot_data.dropna()
stock_zh_spot_data['代码'] = stock_zh_spot_data['代码']
codes_names = np.array(stock_zh_spot_data['代码'])

day = '20100101'
end_date = '20240801'
length = len(codes_names)
all_data = pd.DataFrame([])

for i in tqdm(range(length), desc="Processing items"):
    data_df = ak.stock_zh_a_hist(symbol=codes_names[i], period="daily", start_date=f"{day}", end_date=end_date, adjust="hfq")  # 日度数据，后复权
    data_df['stock_id'] = codes_names[i]
    all_data = all_data.append(data_df)

# 将数据导出为csv文件
all_data.to_csv(os.path.join(f'./data/{day}.csv'), encoding='utf_8_sig')
