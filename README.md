<<<<<<< HEAD
# VI-MLBN: Vision-Informed Mutual Learning Bidirectional Network

## Vi-MLBN Python 环境说明

本项目所需的 Python 环境（含 `PyTorch`、`causal_conv1d`、`mamba_ssm`）**已预先配置完成**，位于：

`$HOME/packages/vi-mlbn/.venv`. 

该环境通过 Singularity 容器构建，与系统隔离，可直接用于训练和推理。

> &#x2753; **为什么必须用容器？**
> 
> 本 HPC 节点的 glibc 版本为 2.28（较旧），而 `mamba-ssm` 等预编译扩展要求 glibc ≥ 2.32。只有在兼容的容器环境中才能正确加载这些 `.so` 文件。

### 🚨 重要提示

- **请勿手动修改** `.venv/` **目录内容**；
- **不要自行运行** `pip install` **或** `conda` **命令**；
- **不要在容器环境之外执行命令**，这既不能使用安装好的环境，又可能污染容器外的环境；
- 除非你明确知道自己在做什么，否则**不要调整环境**。

---

### 🔧 环境损坏？如何恢复

如果环境因任何原因损坏（如 `.venv` 被误删或包导入失败），请在项目根目录下执行以下命令重建：

```bash
make clean          # 彻底清除当前环境
make install        # 重新创建并安装所有依赖
make test-import    # 快速测试安装是否成功
```

整个安装过程约需 1–2 分钟。对于 `make` 有问题可以运行 ``` make help``` 查看说明。

### 🛟 遇到问题？

请勿自行调试底层环境配置，如有需要请联系管理员：<gongch@tongji.edu.cn>.

## 开发环境配置说明

因为本项目的运行依赖完全封装在 Singularity 容器中，VS Code 等本地 IDE 无法直接访问容器内的 Python 解释器和第三方库，导致无法提供代码补全、跳转或类型提示。
为此，我们需要在本地配置一个 dummy Python 环境 —— 它不要求能实际运行项目代码，仅用于向 IDE 提供符号信息（如 `torch`, `numpy` 等模块名）。

我们的 dummy 环境安装在 `./.dev_venv/`, **这不是运行环境**.

### 🛠 创建 dummy 环境

项目提供了 Makefile 目标来自动创建该环境：

```bash
make dummy-venv
```

该命令会：
1. （可选）如果系统支持 `module` 命令（如在 HPC 集群上），会加载 `cuda/12.4` 模块；
2. 使用 `micromamba` 根据 `environment_dev_dummy.yml` 创建一个本地 `conda` 环境到 `./.dev_venv/`；
3. 安装项目所需的两个预编译 wheel 包（`causal_conv1d` 和 `mamba_ssm`）；
4. 安装项目本身及其开发依赖（通过 `pip install ".[train,dev]"`）;
5. 支持在服务器或**本地**安装 (如果通过 git clone 在本地开发).

### 📚 配置 IDE 使用 dummy 环境

以 VS Code 为例：
1. 打开项目根目录；
1. 按 `Ctrl+Shift+P`（或 `Cmd+Shift+P` on macOS），输入并选择 **Python: Select Interpreter**；
1. 选择路径为 `./.dev_venv/bin/python` 的解释器；
1. 此后，IDE 即可正确解析项目依赖，提供代码补全、跳转和类型检查等功能。

### 🚮 清理 dummy 环境

如需删除该环境（例如重建或节省磁盘空间），运行：

```bash
make dummy-clean
```

此命令会安全移除 `./.dev_venv/` 目录。

> ⚠️ 再次强调：dummy 环境仅用于开发辅助，所有实际运行必须在 Singularity 容器内进行。

## 环境依赖详解

本项目依赖以下**非 PyPI 组件**，已预先配置完毕：

- **基础容器**: `pytorch-2.5.1-cuda124-cudnn9-devel.sif`
- **额外包**:
  - `causal_conv1d`: 通过预编译 wheel 安装（见 `./wheels/`）
  - `mamba-ssm`: 通过预编译 wheel 安装（同见 `./wheels/`，要求 `numpy<2.0`）

项目本身的 Python 依赖（如 `timm` 等）由 `pyproject.toml` 管理。

### 🧰 Prerequisites

- [Singularity](https://sylabs.io/guides/4.2/user-guide/) >= 4.2
- `make` (GNU Make)
- Pre-downloaded `.whl` files for:
  - [`causal_conv1d`](https://github.com/Dao-AILab/causal-conv1d)
  - [`mamba_ssm`](https://github.com/state-spaces/mamba)
- A PyTorch 2.5+ CUDA-enabled Singularity image (e.g., `pytorch-2.5.1-cuda124-cudnn9-devel.sif`)

> 💡 The project assumes you already have the `.sif` image and wheels in place (see below).

### 📁 Directory Structure

```text
vi-mlbn/
├── logs
├── Makefile                  # Makefile for environment management
├── pyproject.toml
├── pytorch-2.5.1-cuda124-cudnn9-devel.sif
├── README.md
├── src
    └── ...                   # Source code
└── wheels/
    ├── causal_conv1d-*.whl   # Pre-built wheel files
    └── mamba_ssm-*.whl       # Pre-built wheels
```

> ⚠️ **Important**: Place your `.sif` image and wheels in this directory before running `make`.

### 🔧 Make Targets

| Target | Description |
| ------ | ----------- |
| `make venv` | Create a Python virtual environment in `.venv/` inside the container |
| `make install`	| Install `causal_conv1d` and `mamba_ssm` from local wheels into the venv |
| `make test-import` |	Verify that `torch`, `causal_conv1d`, and `mamba_ssm` import correctly |
| `make shell` | Launch an interactive shell with the venv activated (for debugging) |
| `make clean` | Remove the virtual environment (`rm -rf .venv`) |

All paths we use the in `Makefile` are container-safe and use explicit interpreter paths (`/host/.venv/bin/python`) — no reliance on `PATH` or `activate`.

### 🛠 How It Works

- **Isolation**: A virtual environment is created at `/host/.venv` inside the Singularity container.
- **Reproducibility**: Wheels are installed from the local `wheels/` directory.
- **Determinism**: Every command uses the full path to the venv’s Python interpreter (`$(VENV_PYTHON)`), avoiding PATH ambiguity.
- **HPC Friendly**: Uses `--bind $(CURDIR):/host` for consistent path mapping across login nodes and compute nodes.

## Vision Mamba (Vim) 配置说明

我们使用 [Vision Mamba (Vim)](https://github.com/hustvl/Vim) 和我们的模型 Vi-MLBN 的训练效果做对比。下面是 Vim 的配置说明。

‼️ **注意：**
Vim 的 Github repo 已经克隆在 `$HOME/apps/Vim`。
Vim 依赖 Python 3.10.13, Pytorch 2.1.1 和 CUDA 11.8. 不要使用其他版本，否则可能会出现依赖冲突。

### 安装

> 🚨 **注意：** 虚拟环境 `vim_env` **已经安装完成并可以使用**。运行 `micromamba env list` 确认环境存在。运行 `micromamba activate vim_env` 激活环境。
>
> 以下内容为安装说明，以备之后需要重新安装，或者安装其他版本时使用。

在 HPC 登录节点，使用 micromamba 创建虚拟环境 `vim_env`，并以 uv 管理 Python 依赖。

```bash
# 装载 CUDA 11.8，以备之后编译安装时使用
module load cuda/11.8

# 确认装载成功以及 CUDA 版本
nvcc --version

cd $HOME/apps/Vim

# 用 micromamba 创建虚拟环境 vim_env
micromamba create -n vim_env python=3.10.13 uv=0.9

# 确认环境创建成功
micromamba env list

# 激活环境
micromamba activate vim_env

# 安装 PyTorch 相关依赖
uv pip install \
    torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
    --index-url https://download.pytorch.org/whl/cu118

# 安装 VIM 依赖
uv pip install -r vim/vim_requirements.txt

# 编译安装 causal_conv1d 和 mamba-ssm
uv pip install -e causal-conv1d/ --no-build-isolation
uv pip install -e mamba-1p1p1/ --no-build-isolation
```

## 测试安装

### Smoke Test

执行基本的 import 测试：

```bash
micromamba activate vim_env

python -c "import torch; print('torch version:', torch.__version__)"
python -c "import causal_conv1d; print('causal_conv1d version:', causal_conv1d.__version__)"
python -c "import mamba_ssm; print('mamba_ssm version:', mamba_ssm.__version__)"
PYTHONPATH=/share/home/u23133/apps/Vim python -c "from vim.models_mamba import VisionMamba"
```

### Integration Test

验证真实调度系统（Slurm）下能否跑通。通过 `sbatch` 提交一个小型作业脚本，使用轻量的测试数据。

```bash
sbatch ./slurm/tests/test_vim_dummy.sh
```
=======
# amic-mlbn
>>>>>>> 28a3b6168ea8e625671397b65301efd01b1a4ea1
