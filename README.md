# DL-in-A-share-Trading
本代码使用深度学习模型对中国A股进行训练，从而用于股票交易预测。

# 依赖环境:在开始运行代码之前，请确保已安装以下 Python 库
- scikit-learn
- numpy
- pandas
- matplotlib
- torch
- tqdm
- akshare
# 安装依赖
大部分依赖库可以通过以下命令安装：
  ```bash
  pip install scikit-learn numpy pandas matplotlib tqdm akshare
  ```
# PyTorch 安装
PyTorch 的安装需要根据你的硬件配置（如是否使用 GPU）来选择合适的版本。请访问 PyTorch 官方网站 获取正确的安装命令。

# 使用步骤
- 第一步：数据集构建
- 首先，运行 Building_a_Dataset.py 脚本，这将使用 akshare 库爬取中国A股的相关数据。运行结束后，你将在 data 目录下得到一个 CSV 文件。
  ```bash
  python Building_a_Dataset.py
  ```
# 第二步：模型训练
- 接下来，运行 Training.py 脚本来训练模型。训练过程中，数据集会自动划分为训练集和测试集。你可以使用这些数据集来调整模型结构和参数。训练结束后，模型的权重文件会保存在 weights 文件夹中。
  ```bash
  python Training.py
  ```
# 第三步：模型应用
- 在训练模型并调整完参数后，运行 5_fold_CV.py 脚本进行五折交叉验证。此步骤会进一步优化模型，并在 weights 文件夹中保存最终的模型权重文件，这些权重可以在未来实际应用中使用。
  ```bash
  python 5_fold_CV.py
  ```
