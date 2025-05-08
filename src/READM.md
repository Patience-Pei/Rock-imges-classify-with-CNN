## 代码运行说明

代码运行所需的包如下：
```
torch==2.7.0+cu118
torchaudio==2.7.0+cu118
torchvision==0.22.0+cu118
matplotlib==3.8.4
```
对于 torch 环境的安装，建议在 [PyTorch 官网](https://pytorch.org/get-started/locally/) 中根据指引生成安装 torch 环境的命令安装，其中 CUDA 版本为 11.8。例如 Windows 环境下的安装命令为：
```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

安装完成后直接运行 `main.py` 即可。