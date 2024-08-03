# 模型简介
本项目中的神经网络模型结合了多种深度学习技术，专为中国A股市场的股票交易预测而设计。模型架构融合了全连接层、1D 和 2D 卷积层、以及 LSTM（长短期记忆网络）层，能够处理复杂的时序数据和多维特征。模型的核心设计旨在捕捉股票市场的潜在模式，并生成精准的交易信号。

# 模型架构亮点
- 全连接层：通过多个全连接层与批归一化和 Dropout 技术相结合，提高了模型的非线性表达能力，并有效防止过拟合。
- 1D 卷积层：提取时间序列数据中的局部特征，有效处理金融数据中的高频波动。
- 2D 卷积层：进一步挖掘二维特征间的相关性，为复杂的数据关系提供更深层次的洞察。
- LSTM 层：利用 LSTM 处理序列数据中的长期依赖性，特别适合捕捉股市中的趋势和周期性变化。
- 模型初始化：采用 He 初始化方法（Kaiming 正态初始化），确保模型在训练初期具备良好的收敛性。
- 这个模型经过精心设计，能够从多维数据中提取信息，并通过复杂的非线性组合，生成有助于实际交易的预测结果。

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
