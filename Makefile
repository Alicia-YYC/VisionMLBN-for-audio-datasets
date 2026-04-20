# Makefile for Vision-MLBN (Singularity + .venv workflow)

SHELL := /bin/bash
IMAGE := pytorch-2.5.1-cuda124-cudnn9-devel.sif
VENV := .venv
VENV_PYTHON := /host/$(VENV)/bin/python
WHEEL_DIR := wheels

# 自动匹配 wheel 文件（支持版本变化）
CAUSAL_WHEEL := $(wildcard $(WHEEL_DIR)/causal_conv1d-*.whl)
MAMBA_WHEEL := $(wildcard $(WHEEL_DIR)/mamba_ssm-*.whl)

# Dummy development environment setup for IDE support
DUMMY_ENV := .dev_venv
DUMMY_ENV_PYTHON := $(DUMMY_ENV)/bin/python

# 检查 wheel 是否存在
ifeq ($(strip $(CAUSAL_WHEEL)),)
$(error No causal_conv1d wheel found in $(WHEEL_DIR)/)
endif
ifeq ($(strip $(MAMBA_WHEEL)),)
$(error No mamba_ssm wheel found in $(WHEEL_DIR)/)
endif

.PHONY: help venv install test-import shell clean dummy-venv dummy-clean

help:
	@echo "Vision-MLBN Development Workflow"
	@echo "--------------------------------"
	@echo "  venv          - Create .venv inside container (with --system-site-packages)"
	@echo "  install       - Install causal_conv1d and mamba_ssm from local wheels"
	@echo "  test-import   - Run non-blocking import tests for torch/causal/mamba"
	@echo "  shell         - Enter interactive shell with venv activated"
	@echo "  clean         - Remove .venv and lock files"
	@echo "  dummy-venv    - Create dummy development environment for IDE support"

# 创建虚拟环境（关键：继承系统包以复用容器中的 PyTorch）
venv:
	@echo "==> Creating isolated virtual environment..." && \
	singularity exec \
		--bind $(CURDIR):/host:rw \
		"$(IMAGE)" \
		bash -c "\
			python3.11 -m venv /host/$(VENV) --system-site-packages --without-pip && \
			$(VENV_PYTHON) -m ensurepip --upgrade && \
			$(VENV_PYTHON) -m pip install --upgrade pip"

# 安装两个本地 wheel（pip 会自动解析并安装其依赖，如 numpy<2.0）
install: venv
	@echo "==> Installing causal_conv1d and mamba_ssm from wheels..." && \
	singularity exec \
		--bind $(CURDIR):/host:rw \
		--bind $(CURDIR)/$(WHEEL_DIR):/mnt/wheels:ro \
		"$(IMAGE)" bash -c "\
			$(VENV_PYTHON) -m pip install --no-user \
				/mnt/wheels/$(notdir $(CAUSAL_WHEEL)) \
				/mnt/wheels/$(notdir $(MAMBA_WHEEL)) \
			$(VENV_PYTHON) -m pip install -e ."

# 运行非阻断式 import 测试（即使失败也继续，便于查看所有错误）
test-import:
	@echo "==> Running comprehensive import tests (errors shown but won't stop execution)..." && \
	singularity exec \
		--bind $(CURDIR):/host:rw \
		"$(IMAGE)" bash -c "\
			( $(VENV_PYTHON) -c \"import torch; print('✅ Torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'GPU Available:', torch.cuda.is_available())\" || echo '❌ Failed to import torch' ) && \
			( $(VENV_PYTHON) -c \"import causal_conv1d; print('✅ causal_conv1d version:', causal_conv1d.__version__)\" || echo '❌ Failed to import causal_conv1d' ) && \
			( $(VENV_PYTHON) -c \"import mamba_ssm; print('✅ mamba-ssm version:', mamba_ssm.__version__); print('✅ Mamba-SSM imported successfully')\" || echo '❌ Failed to import mamba_ssm' ) && \
			echo '✅ Test completed. Check above for any import failures.'"

# 交互式 shell（用于手动调试）
shell: venv
	@echo "==> Entering interactive shell with .venv activated..." && \
	singularity exec \
		--bind $(CURDIR):/host:rw \
		--bind $(CURDIR)/$(WHEEL_DIR):/mnt/wheels:ro \
		"$(IMAGE)" bash -c "source /host/$(VENV)/bin/activate && exec bash"

# 清理
clean:
	@echo "==> Cleaning up..." && \
	rm -rf $(VENV) requirements*.lock.txt

# 创建 dummy 开发环境（仅用于 IDE 支持）
dummy-venv:
	@set -e; \
	if command -v module >/dev/null 2>&1; then \
		echo "Loading CUDA 12.4 module..."; \
		module purge; \
		module load cuda/12.4; \
	fi; \
	nvcc --version; \
	echo "Creating dummy environment in $(DUMMY_ENV)..."; \
	micromamba create -y -p $(PWD)/$(DUMMY_ENV) -f environment_dev_dummy.yml \
	$(DUMMY_ENV_PYTHON) -m pip install --no-user $(WHEEL_DIR)/causal_conv1d-*.whl $(WHEEL_DIR)/mamba_ssm-*.whl \
	$(DUMMY_ENV_PYTHON) -m pip install ".[train,dev]"

dummy-clean:
	@echo "Cleaning up dummy environment in $(DUMMY_ENV)..."
	micromamba env remove -y -p $(PWD)/$(DUMMY_ENV)
