# HSC-Mamba

## 环境配置

Python 3.10、PyTorch 2.1.1、CUDA 12.1。

```bash
conda create -n hsc-mamba python=3.10 -y
conda activate hsc-mamba
conda install -c nvidia/label/cuda-12.1.1 cuda-toolkit -y

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

python -m pip install torch==2.1.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install causal-conv1d==1.1.3.post1 --no-build-isolation
python -m pip install mamba-ssm==1.1.3.post1 --no-build-isolation
```

在包含 `run.py` 的项目根目录执行以下命令。数据集默认位于 `/Featurize/data`，子目录为 `wsj0-2mix`、`Libri2Mix`、`wham` 和 `whamr`，均使用 `wav8k/min` 数据。首次运行前生成数据路径清单：

```bash
bash data_path/generate_all_scp.sh
```

## 训练

```bash
# WSJ0-2Mix
python -u run.py --dataset wsj0-2mix --engine-mode train

# Libri2Mix
python -u run.py --dataset libri2mix --engine-mode train

# WHAM!
python -u run.py --dataset wham --engine-mode train

# WHAMR!
python -u run.py --dataset whamr --engine-mode train
```

## 测试

测试使用验证集 SI-SNRi 最高的 5 个 checkpoint 的 EMA 参数平均结果，需要先完成训练。

```bash
# WSJ0-2Mix
python -u run.py --dataset wsj0-2mix --engine-mode test

# Libri2Mix
python -u run.py --dataset libri2mix --engine-mode test

# WHAM!
python -u run.py --dataset wham --engine-mode test

# WHAMR!
python -u run.py --dataset whamr --engine-mode test
```

需要保存分离音频时，将 `--engine-mode test` 改为 `--engine-mode test_save`。
