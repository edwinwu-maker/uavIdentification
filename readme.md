# 无人机识别

## 数据集

- DroneRFa:
  下载链接 [DroneRFa](https://china.scidb.cn/download?fileId=c403fc76444e4b9989e4f3ff570f3b3d&traceId=e1937db6-783a-47dd-9df2-16c869a5dc33)
  数据集说明文档： DroneRFa：用于侦测低空无人机的大规模无人机射频信号数据集.pdf

## 项目目录结构

AGENTS.md                         # 代码代理协作规则和项目运行约定  
readme.md                         # 项目说明文档  
docs/  
└── DroneRFa：用于侦测低空无人机的大规模无人机射频信号数据集.pdf  # 数据集说明文档  
src/  
├── __init__.py                   # src包初始化文件  
├── data/  
│   ├── __init__.py               # data包初始化文件  
│   ├── droneRFa_dataset.py       # 原始.mat IQ数据Dataset，负责样本切片、双通道IQ组装和标签解析  
│   ├── spectrogram_dataset.py    # 预计算.h5频谱图Dataset，按样本索引懒加载STFT和标签  
│   └── transforms.py             # IQ到STFT频谱图的数据预处理变换  
├── models/  
│   ├── __init__.py               # models包初始化文件  
│   └── resnet.py                 # 适配双通道频谱输入的ResNet-18模型定义  
├── scripts/  
│   ├── __init__.py               # scripts包初始化文件  
│   ├── generate_cpp_png.py       # 从CPP/FAM .h5文件生成双通道CPP/FAM图片  
│   ├── generate_png.py           # 从STFT .h5文件生成双通道频谱图PNG  
│   ├── plot_fam_demo.py          # 生成合成信号的时域、频域和FAM可视化示例  
│   ├── precompute_cpp_h5.py      # 将原始.mat IQ数据预计算为CPP/FAM .h5文件  
│   └── precompute_h5.py          # 将原始.mat IQ数据预计算为STFT .h5文件  
├── utils/  
│   ├── __init__.py               # utils包初始化文件  
│   ├── fam.py                    # CPU/NumPy版本FAM/SCF计算工具  
│   ├── fam_grid.py               # FAM稀疏点到二维CPP/FAM网格的聚合和分段合并工具  
│   ├── fam_plot.py               # FAM/CPP热力图、三维曲面和时频图绘制工具  
│   ├── fam_torch.py              # PyTorch版本FAM/SCF计算工具，支持CUDA、MPS或CPU  
│   ├── logger.py                 # 全局日志工具  
│   ├── plot_utils.py             # 通用时域、频域和STFT可视化工具  
│   └── synthetic_signal.py       # FAM示例使用的BPSK、加噪BPSK和白噪声信号生成器  
├── test_spec.py                  # 加载训练模型并在预计算.h5数据上测试  
├── train.py                      # 直接读取原始.mat IQ数据并在线转换STFT进行训练  
├── train_spec.py                 # 基于预计算STFT .h5数据的单GPU训练脚本  
└── train_spec_ddp.py             # 基于预计算STFT .h5数据的多GPU DDP训练脚本  
