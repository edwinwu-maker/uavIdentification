# 无人机识别

## 数据集

- DroneRFa:
  下载链接 [DroneRFa](https://china.scidb.cn/download?fileId=c403fc76444e4b9989e4f3ff570f3b3d&traceId=e1937db6-783a-47dd-9df2-16c869a5dc33)
  数据集说明文档： DroneRFa：用于侦测低空无人机的大规模无人机射频信号数据集.pdf

## 项目目录结构

src/
├── data/
│   ├── droneRFa_dataset.py       # 原始.mat IQ数据Dataset
│   ├── spectrogram_dataset.py    # STFT矩阵.npy读取Dataset
│   └── transforms.py             # 数据预处理变换
├── models/
│   └── resnet.py                 # ResNet模型定义
├── scripts/
│   ├── generate_spectrograms.py  # 生成时频图PNG
│   └── precompute_spectrograms.py # 预计算STFT存npy
├── utils/
│   ├── logger.py                 # 全局日志工具
│   ├── plot_utils.py             # 绘图可视化工具
│   └── signal_process.py         # 信号处理通用函数
├── train_spec.py                 # 基于STFT矩阵的训练脚本
└── train.py
