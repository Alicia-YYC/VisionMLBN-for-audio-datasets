"""VisionMLBN在Tiny-ImageNet-200数据集上的训练脚本"""

from datetime import datetime
import logging
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from jsonargparse import ActionConfigFile, ArgumentParser
from timm.data.mixup import Mixup
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils.model_ema import ModelEmaV2
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter

from vision_mlbn.VisionMLBN import vim_small_patch16_224_mlbn, vim_tiny_patch16_224_mlbn

logger = logging.getLogger("train_tiny_imagenet")


def setup_ddp(rank, world_size):
    """初始化 DDP 进程组"""
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")

    # 初始化进程组
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)

    # 设置当前进程使用的 GPU
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """清理 DDP 进程组"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """判断是否为主进程"""
    return not dist.is_initialized() or dist.get_rank() == 0


class RepeatAugSampler(torch.utils.data.Sampler):
    """Repeated Augmentation Sampler (DeiT)

    每个样本在一个epoch中重复num_repeats次，每次使用不同的增强。
    这可以提升数据利用效率，特别是在小数据集上。

    参考: DeiT - Training data-efficient image transformers
    """

    def __init__(self, dataset, num_repeats=3, shuffle=True, seed=0):
        self.dataset = dataset
        self.num_repeats = num_repeats
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.num_samples = len(dataset) * num_repeats

    def __iter__(self):
        # DeiT正确实现: 每个repeat独立shuffle，让同一样本在epoch内分散出现
        repeated_indices = []

        for repeat_idx in range(self.num_repeats):
            if self.shuffle:
                g = torch.Generator()
                # 每个repeat使用不同的seed，确保顺序不同
                g.manual_seed(self.seed + self.epoch * 1000 + repeat_idx * 100)
                indices = torch.randperm(len(self.dataset), generator=g).tolist()
            else:
                indices = list(range(len(self.dataset)))
            repeated_indices.extend(indices)

        return iter(repeated_indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedRepeatAugSampler(torch.utils.data.Sampler):
    """DDP + Repeated Augmentation Sampler

    结合 DistributedSampler 和 RepeatAugSampler 的功能。
    每个样本重复 num_repeats 次，然后均匀分配到各个 GPU。
    """

    def __init__(self, dataset, num_repeats=3, num_replicas=None, rank=None, shuffle=True, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_repeats = num_repeats
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed

        # 总样本数 = 原始样本数 × 重复次数
        self.total_size = len(self.dataset) * self.num_repeats
        # 每个 GPU 分配的样本数（向上取整以确保覆盖所有样本）
        self.num_samples = int(math.ceil(self.total_size / self.num_replicas))
        # 实际总大小（可能需要 padding）
        self.padded_total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # 生成重复的索引
        repeated_indices = []

        for repeat_idx in range(self.num_repeats):
            if self.shuffle:
                # 每个 repeat 使用不同的随机顺序
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch * 1000 + repeat_idx * 100)
                indices = torch.randperm(len(self.dataset), generator=g).tolist()
            else:
                indices = list(range(len(self.dataset)))
            repeated_indices.extend(indices)

        # Padding 到可以被 num_replicas 整除
        if len(repeated_indices) < self.padded_total_size:
            # 用重复的索引填充
            repeated_indices += repeated_indices[: (self.padded_total_size - len(repeated_indices))]

        # 为当前 rank 分配子集
        indices = repeated_indices[self.rank : self.padded_total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def _sanitize_filename(name: str) -> str:
    """
    将文件名称中的非法字符替换为'_'

    Parameters:
    ----------
    name: 原始文件名称

    Returns:
    --------
    output: 新文件名称
    """
    if not name:
        return "train"
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


def configure_logging(log_dir: str, job_name: str = "", job_id: str = ""):
    """配置日志到控制台和可选文件，返回(stdout_path, stderr_path)

    Parameters:
    ----------
    log_dir: 日志保存目录
    job_name: 任务名称
    job_id: 任务id

    Returns:
    --------
    stdout_path: 标准输出路径
    stderr_path: 标准错误输出路径
    """

    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    stdout_path = ""
    stderr_path = ""
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        base_name = _sanitize_filename(job_name)
        base_name = (
            "_".join([base_name, str(job_id)]) if job_id else "_".join([base_name, time.strftime("%Y%m%d_%H%M%S")])
        )
        stdout_path = os.path.join(log_dir, f"{base_name}.out")
        stderr_path = os.path.join(log_dir, f"{base_name}.err")

        file_handler = logging.FileHandler(stdout_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        error_handler = logging.FileHandler(stderr_path)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)
    else:
        logger.error("日志保存目录不存在, 请检查日志保存目录是否正确.")
        raise ValueError("日志保存目录不存在, 请检查日志保存目录是否正确.")
    return stdout_path, stderr_path


def seed_worker(worker_id):
    """确保DataLoader子进程拥有可复现的随机状态"""

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed):
    """设置全局随机种子，返回有效种子（None 表示未固定）"""

    if seed is None:
        logger.warning("未设置固定随机种子，训练结果可能无法完全复现。")
        return None

    logger.info(f"设置随机种子: {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed


class AverageMeter:
    """计算并存储平均值和当前值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # 如果是tensor，立即转为标量（恢复原始简单逻辑）
        if torch.is_tensor(val):
            val = val.item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


class SymmetryLossWrapper(nn.Module):
    """包装现有损失函数，添加对称损失"""

    def __init__(self, base_criterion, symmetry_weight=0.01):
        super().__init__()
        self.base_criterion = base_criterion
        self.symmetry_weight = symmetry_weight

    def forward(self, outputs, target, model=None, **kwargs):
        # 基础损失 - 直接调用，兼容所有标准损失函数
        base_loss = self.base_criterion(outputs, target, **kwargs)

        # 对称损失 - 快速路径：如果没有model或权重为0，直接返回基础损失
        if model is None or self.symmetry_weight == 0.0:
            return base_loss

        # 计算对称损失（带异常保护）
        try:
            symmetry_loss = model.encoder.compute_weighted_symmetry_loss()
            return base_loss + self.symmetry_weight * symmetry_loss
        except Exception:
            # 对称损失计算失败时，返回基础损失
            return base_loss


class MixedLoss(nn.Module):
    """混合损失函数，结合多个损失函数以提高模型性能"""

    def __init__(self, num_classes, smoothing=0.1, aux_weight=0.2):
        super().__init__()
        self.smoothing = smoothing
        self.aux_weight = aux_weight
        self.num_classes = num_classes

        # 主损失：标签平滑交叉熵
        self.main_criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)

        # 辅助损失1：Focal Loss - 处理类别不平衡
        self.gamma = 2.0  # focal loss的聚焦参数

        # 辅助损失2：中心损失 - 增强特征区分度
        self.center_weight = 0.1
        self.centers = nn.Parameter(torch.randn(num_classes, 512))  # 类别中心

    def focal_loss(self, x, target):
        """计算Focal Loss"""
        ce_loss = F.cross_entropy(x, target, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

    def center_loss(self, features, target):
        """计算中心损失"""
        batch_size = features.size(0)
        features = F.normalize(features, p=2, dim=1)  # L2标准化
        centers = F.normalize(self.centers, p=2, dim=1)  # L2标准化

        # 计算每个样本到其对应类别中心的距离
        target_centers = centers[target]
        center_distances = torch.sum((features - target_centers) ** 2, dim=1)

        return center_distances.mean()

    def forward(self, outputs, target, features=None):
        """
        outputs: 模型输出
        target: 目标标签
        features: 特征向量（可选）
        """
        if isinstance(outputs, tuple):
            # 处理多输出情况（主输出和辅助输出）
            main_output = outputs[0]
            aux_outputs = outputs[1:]

            # 主损失
            main_loss = self.main_criterion(main_output, target)

            # Focal Loss
            focal = self.focal_loss(main_output, target)

            # 辅助损失
            aux_loss = sum(self.main_criterion(aux, target) for aux in aux_outputs)

            # 如果提供了特征向量，添加中心损失
            if features is not None:
                center = self.center_loss(features, target)
                total_loss = main_loss + self.aux_weight * aux_loss + 0.5 * focal + self.center_weight * center
            else:
                total_loss = main_loss + self.aux_weight * aux_loss + 0.5 * focal

        else:
            # 单输出情况
            main_loss = self.main_criterion(outputs, target)
            focal = self.focal_loss(outputs, target)

            if features is not None:
                center = self.center_loss(features, target)
                total_loss = main_loss + 0.5 * focal + self.center_weight * center
            else:
                total_loss = main_loss + 0.5 * focal

        return total_loss


class EarlyStopping:
    """早停机制 - 防止过拟合，当验证集长期未提升时停止训练"""

    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None

    def __call__(self, val_score):
        """返回 (should_stop, should_reduce_lr)，当前策略不再自动降LR"""
        if self.best_score is None:
            self.best_score = val_score
            return False, False
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True, False
        else:
            self.best_score = val_score
            self.counter = 0
        return False, False


def get_tiny_imagenet_data_loaders(
    dataset_path,
    batch_size=128,
    num_workers=4,
    use_strong_augmentation=False,
    seed=None,
    repeated_aug=1,
    use_ddp=False,
    input_size=224,
):
    """获取Tiny-ImageNet-200数据加载器

    Args:
        use_ddp: 是否使用 DDP，若为 True 则使用 DistributedSampler
    """
    # 只在主进程打印（DDP 模式下）或单卡模式打印
    is_main = not use_ddp or not dist.is_initialized() or dist.get_rank() == 0

    if is_main:
        logger.info("ImageFolder spectrogram 数据集...")

    # spectrogram baseline
    train_transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
    )

    # 加载数据集
    train_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(dataset_path, "train"), transform=train_transform
    )

    val_dataset = torchvision.datasets.ImageFolder(root=os.path.join(dataset_path, "val"), transform=val_transform)

    # 创建数据加载器
    worker_init_fn = seed_worker if seed is not None else None

    train_generator = None
    val_generator = None
    if seed is not None:
        train_generator = torch.Generator()
        train_generator.manual_seed(seed)

        val_generator = torch.Generator()
        val_generator.manual_seed(seed + 1)

    # Repeated Augmentation (DeiT技巧) 与 DDP
    train_sampler = None
    val_sampler = None
    shuffle_train = True

    if use_ddp:
        # DDP 模式
        if repeated_aug > 1:
            # DDP + Repeated Aug: 使用 DistributedRepeatAugSampler
            train_sampler = DistributedRepeatAugSampler(
                train_dataset,
                num_repeats=repeated_aug,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=True,
                seed=seed if seed is not None else 0,
            )
            if is_main:
                logger.info(f"启用 DDP + Repeated Augmentation: 每个样本重复 {repeated_aug} 次")
                logger.info(
                    f"总有效样本数: {len(train_dataset)} × {repeated_aug} = {len(train_dataset) * repeated_aug}"
                )
                logger.info(f"每个 GPU 分配样本数: ~{len(train_dataset) * repeated_aug // dist.get_world_size()}")
        else:
            train_sampler = DistributedSampler(
                train_dataset,
                shuffle=True,
                seed=seed if seed is not None else 0,
            )

        val_sampler = DistributedSampler(
            val_dataset,
            shuffle=False,
        )
        shuffle_train = False
        if is_main and repeated_aug <= 1:
            logger.info(
                f"启用 DDP 模式: DistributedSampler (world_size={dist.get_world_size()}, rank={dist.get_rank()})"
            )
    elif repeated_aug > 1:
        # 单卡 + Repeated Aug
        train_sampler = RepeatAugSampler(
            train_dataset,
            num_repeats=repeated_aug,
            shuffle=True,
            seed=seed if seed is not None else 0,
        )
        shuffle_train = False  # 使用sampler时不能shuffle
        if is_main:
            logger.info(f"启用 Repeated Augmentation: 每个样本重复 {repeated_aug} 次")
            logger.info(f"有效训练样本数: {len(train_dataset)} × {repeated_aug} = {len(train_dataset) * repeated_aug}")

    # DataLoader配置
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
        "worker_init_fn": worker_init_fn,
    }

    # 添加提速配置
    if num_workers > 0:
        import multiprocessing

        # 强制使用fork避免worker重新初始化TensorFlow (大幅提速)
        loader_kwargs["multiprocessing_context"] = multiprocessing.get_context("fork")
        # persistent_workers: 保持worker进程存活，避免每个epoch重启 (~10-20%提速)
        # 注意：长时间训练可能有内存累积，但性能收益显著，可通过周期性清理缓存缓解
        loader_kwargs["persistent_workers"] = True
        # prefetch_factor: 减少预加载数量，降低内存压力 (原4→2)
        loader_kwargs["prefetch_factor"] = 2
        if is_main:
            logger.info(f"DataLoader 提速配置: fork={True}, persistent_workers={True}, prefetch_factor=2")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train if train_sampler is None else False,
        sampler=train_sampler,
        drop_last=True,
        generator=train_generator if train_sampler is None else None,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False if val_sampler is None else False,
        sampler=val_sampler,
        generator=val_generator if val_sampler is None else None,
        **loader_kwargs,
    )

    if is_main:
        logger.info(f"训练集: {len(train_dataset)} 张图片, {len(train_dataset.classes)} 个类别")
        logger.info(f"验证集: {len(val_dataset)} 张图片, {len(val_dataset.classes)} 个类别")
        logger.info(f"输入尺寸: {input_size} x {input_size}")
        logger.info("训练集变换: Resize -> ToTensor -> Normalize")
        logger.info("验证集变换: Resize -> ToTensor -> Normalize")
        logger.info("归一化参数: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")
        logger.info(f"类别映射: {train_dataset.class_to_idx}")

    return train_loader, val_loader, len(train_dataset.classes)


def create_model(model_name, num_classes, img_size=224, drop_path_rate=0.1):
    """创建模型

    Args:
        model_name: 模型名称 ('tiny' 或 'small')
        num_classes: 分类类别数
        drop_path_rate: Stochastic Depth的drop path rate (DeiT技巧)
    """
    if model_name == "tiny":
        model = vim_tiny_patch16_224_mlbn(
            num_classes=num_classes,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
        )
        logger.info(f"创建VisionMLBN-Tiny模型 ({img_size}x{img_size}, drop_path_rate={drop_path_rate})")
    elif model_name == "small":
        model = vim_small_patch16_224_mlbn(
            num_classes=num_classes,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
        )
        logger.info(f"创建VisionMLBN-Small模型 ({img_size}x{img_size}, drop_path_rate={drop_path_rate})")
    else:
        raise ValueError(f"不支持的模型: {model_name}")

    return model


def train_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    mixup_fn,
    model_ema,
    device,
    epoch,
    max_grad_norm=None,
    gradient_accumulation_steps=1,
    use_amp=False,
    amp_dtype=None,
    scaler=None,
):
    """训练一个epoch"""
    model.train()

    losses = AverageMeter()
    symmetry_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    optimizer.zero_grad()

    # 移除进度条，减少输出
    for batch_idx, (data, target) in enumerate(train_loader):
        # non_blocking=True: 异步传输，CPU→GPU传输与计算重叠 (~5-10%提速)
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # 将数据增强后的图片沿主对角线翻转后加入训练
        # 注释：在微调阶段此功能未带来明显收益，暂时禁用以加快训练速度
        # diag_data = data.transpose(-1, -2).contiguous()
        # if target.dim() == 1:
        #     diag_target = target
        # else:
        #     diag_target = target.clone()
        # data = torch.cat([data, diag_data], dim=0)
        # target = torch.cat([target, diag_target], dim=0)

        if mixup_fn is not None:
            data, target = mixup_fn(data, target)

        amp_cast_dtype = amp_dtype if amp_dtype is not None else torch.float16
        with autocast(device_type="cuda", enabled=use_amp, dtype=amp_cast_dtype):
            output = model(data)
            if isinstance(criterion, SymmetryLossWrapper):
                loss = criterion(output, target, model=model)
            else:
                loss = criterion(output, target)

        loss_to_backward = loss / gradient_accumulation_steps
        if scaler is not None and use_amp:
            scaler.scale(loss_to_backward).backward()
        else:
            loss_to_backward.backward()

        # 计算准确率（mixup情形下用硬标签近似：one-hot取argmax）
        hard_target = target.argmax(dim=1) if target.dim() == 2 else target
        prec1, prec5 = accuracy(output, hard_target, topk=(1, 5))

        # 延迟同步：使用 detach() 避免每个 batch 都同步
        losses.update(loss, data.size(0))
        top1.update(prec1[0], data.size(0))
        top5.update(prec5[0], data.size(0))

        # 梯度累积步数达到时更新参数
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if scaler is not None and use_amp:
                if max_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            if model_ema is not None:
                model_ema.update(model)
            optimizer.zero_grad()

    # 训练epoch结束，清理梯度和临时缓存
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return losses.avg, top1.avg, top5.avg


def validate(model, val_loader, criterion, device, use_amp=False, amp_dtype=None):
    """验证模型"""
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        # 移除进度条，减少输出
        for data, target in val_loader:
            # non_blocking=True: 异步传输
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            amp_cast_dtype = amp_dtype if amp_dtype is not None else torch.float16
            with autocast(device_type="cuda", enabled=use_amp, dtype=amp_cast_dtype):
                output = model(data)
                if isinstance(criterion, SymmetryLossWrapper):
                    loss = criterion(output, target, model=model)
                else:
                    loss = criterion(output, target)

            # 计算准确率
            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            losses.update(loss, data.size(0))
            top1.update(prec1[0], data.size(0))
            top5.update(prec5[0], data.size(0))

    return losses.avg, top1.avg, top5.avg


def validate_with_tta(
    model,
    val_loader,
    criterion,
    device,
    use_amp=False,
    amp_dtype=None,
    tta_transforms=None,
):
    """
    Test-Time Augmentation (TTA) 验证

    原理：对每张图片进行多次增强变换，将多个预测结果进行平均，通常能提升0.5-2%准确率。

    Args:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备
        use_amp: 是否使用混合精度
        amp_dtype: 混合精度数据类型
        tta_transforms: TTA变换列表，None则使用默认变换

    Returns:
        losses.avg, top1.avg, top5.avg
    """
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # 默认TTA变换：原图 + 水平翻转
    if tta_transforms is None:
        tta_transforms = [
            lambda x: x,  # 原图
            lambda x: torch.flip(x, dims=[3]),  # 水平翻转
        ]

    with torch.no_grad():
        for data, target in val_loader:
            # non_blocking=True: 异步传输
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            amp_cast_dtype = amp_dtype if amp_dtype is not None else torch.float16

            # 对每个TTA变换进行预测，然后平均
            outputs = []
            with autocast(device_type="cuda", enabled=use_amp, dtype=amp_cast_dtype):
                for transform in tta_transforms:
                    transformed_data = transform(data)
                    output = model(transformed_data)
                    outputs.append(F.softmax(output, dim=1))

                # 平均所有预测
                avg_output = torch.stack(outputs, dim=0).mean(dim=0)
                # 转换为logits用于损失计算
                output_logits = torch.log(avg_output + 1e-8)
                if isinstance(criterion, SymmetryLossWrapper):
                    loss = criterion(output_logits, target, model=model)
                else:
                    loss = criterion(output_logits, target)

            # 计算准确率
            prec1, prec5 = accuracy(avg_output, target, topk=(1, 5))

            losses.update(loss, data.size(0))
            top1.update(prec1[0], data.size(0))
            top5.update(prec5[0], data.size(0))

    return losses.avg, top1.avg, top5.avg


class SWAModel:
    """
    Stochastic Weight Averaging (SWA)

    原理：在训练后期，对多个epoch的模型权重进行等权平均，找到更平坦的最优解，
    通常能提升0.5-1.5%泛化性能。

    与EMA区别：
    - EMA: 指数衰减平均，全程启用
    - SWA: 等权重平均，仅在训练后期启用，通常配合周期性高LR

    使用方法：
        swa = SWAModel(model)
        for epoch in range(swa_start, total_epochs):
            train_one_epoch(model)
            swa.update(model)
        swa.finalize()  # 更新BN统计量
        # 使用 swa.model 进行推理
    """

    def __init__(self, model, device=None):
        """
        初始化SWA模型

        Args:
            model: 原始模型
            device: 设备
        """
        import copy

        self.model = copy.deepcopy(model)
        self.device = device
        self.n_averaged = 0

        if device is not None:
            self.model.to(device)

    def update(self, model):
        """
        更新SWA平均权重

        Args:
            model: 当前训练的模型
        """
        with torch.no_grad():
            for swa_param, model_param in zip(self.model.parameters(), model.parameters(), strict=False):
                # 等权重平均: new_avg = (old_avg * n + new_val) / (n + 1)
                swa_param.data.mul_(self.n_averaged).add_(model_param.data).div_(self.n_averaged + 1)
        self.n_averaged += 1

    def finalize(self, train_loader, device, use_amp=False, amp_dtype=None):
        """
        完成SWA：更新BatchNorm统计量

        SWA平均后的权重会导致BN层的running_mean/running_var不准确，
        需要在训练数据上重新计算。

        Args:
            train_loader: 训练数据加载器
            device: 设备
            use_amp: 是否使用混合精度
            amp_dtype: 混合精度类型
        """
        # 检查模型是否包含BN层
        has_bn = any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) for m in self.model.modules())

        if not has_bn:
            return

        # 重置BN统计量
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.reset_running_stats()
                m.momentum = None  # 使用累积平均

        # 在训练数据上更新BN统计量
        self.model.train()
        amp_cast_dtype = amp_dtype if amp_dtype is not None else torch.float16

        with torch.no_grad():
            for data, _ in train_loader:
                data = data.to(device, non_blocking=True)
                with autocast(device_type="cuda", enabled=use_amp, dtype=amp_cast_dtype):
                    self.model(data)

        self.model.eval()
        logger.info(f"SWA: 已更新BN统计量 (平均了 {self.n_averaged} 个模型)")


def get_tta_transforms(level="light", input_size=224):
    """
    获取TTA变换列表

    Args:
        level: 'light' (2x, 快速), 'medium' (4x), 'heavy' (8x, 最准但慢)
        input_size: 模型输入尺寸，用于确保缩放后恢复原尺寸

    Returns:
        变换函数列表
    """

    # 缩放+恢复尺寸的辅助函数
    def scale_and_resize(x, scale):
        scaled = F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)
        return F.interpolate(scaled, size=(input_size, input_size), mode="bilinear", align_corners=False)

    if level == "light":
        # 2x: 原图 + 水平翻转
        return [
            lambda x: x,
            lambda x: torch.flip(x, dims=[3]),
        ]
    elif level == "medium":
        # 4x: 原图 + 水平翻转 + 两种轻微缩放（模拟不同距离）
        return [
            lambda x: x,
            lambda x: torch.flip(x, dims=[3]),
            lambda x: scale_and_resize(x, 0.9),  # 稍微缩小再放大
            lambda x: scale_and_resize(x, 1.1),  # 稍微放大再缩小
        ]
    elif level == "heavy":
        # 8x: 更多变换组合
        return [
            lambda x: x,
            lambda x: torch.flip(x, dims=[3]),  # 水平翻转
            lambda x: torch.flip(x, dims=[2]),  # 垂直翻转
            lambda x: torch.flip(torch.flip(x, dims=[3]), dims=[2]),  # 水平+垂直
            lambda x: scale_and_resize(x, 0.85),
            lambda x: scale_and_resize(x, 1.15),
            lambda x: torch.flip(scale_and_resize(x, 0.9), dims=[3]),
            lambda x: torch.flip(scale_and_resize(x, 1.1), dims=[3]),
        ]
    else:
        return [lambda x: x]


def accuracy(output, target, topk=(1,)):
    """计算top-k准确率；当k大于类别数时自动截断"""
    with torch.no_grad():
        num_classes = output.size(1)
        adjusted_topk = tuple(min(k, num_classes) for k in topk)
        maxk = max(adjusted_topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in adjusted_topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate_cosine(
    optimizer,
    epoch,
    base_lr,
    warmup_epochs,
    plateau_epochs,
    total_epochs,
    min_lr,
    mid_lr,
    mid_lr_epoch,
):
    """线性warmup + plateau + 线性下探 + cosine退火(按epoch)"""

    plateau_end = warmup_epochs + plateau_epochs
    mid_lr_epoch = max(plateau_end + 1, mid_lr_epoch)
    clamped_mid = min(base_lr, max(min_lr, mid_lr))

    if epoch < warmup_epochs:
        lr = base_lr * float(epoch + 1) / float(max(1, warmup_epochs))
    elif epoch < plateau_end:
        lr = base_lr
    elif epoch < mid_lr_epoch:
        span = max(1, mid_lr_epoch - plateau_end)
        progress = (epoch - plateau_end) / span
        lr = base_lr - (base_lr - clamped_mid) * progress
    else:
        effective_epochs = max(1, total_epochs - mid_lr_epoch)
        t = (epoch - mid_lr_epoch) / float(effective_epochs)
        t = min(max(t, 0.0), 1.0)
        lr = min_lr + 0.5 * (clamped_mid - min_lr) * (1.0 + math.cos(math.pi * t))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def main():
    parser = ArgumentParser(description="VisionMLBN Tiny-ImageNet-200 Training")
    parser.add_argument(
        "--config", action=ActionConfigFile, help="YAML/JSON 配置文件路径。命令行参数会覆盖配置文件中的同名字段。"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tiny",
        choices=["tiny", "small"],
        help="模型大小 (default: tiny)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (default: 42, 设为负值则不固定)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Tiny-ImageNet-200数据集路径.",
    )
    parser.add_argument(
    "--input_size",
    type=int,
    default=224,
    help="输入图像尺寸 (default: 224)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="批次大小 (default: 64, 减小批次大小增加正则化)",
    )
    parser.add_argument("--epochs", type=int, default=200, help="训练轮数 (default: 200)")
    parser.add_argument("--lr", type=float, default=5e-4, help="学习率 (AdamW, default: 5e-4)")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="最小学习率 (cosine 终点)")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="权重衰减 (AdamW)")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="标签平滑系数 (default: 0.0)")
    parser.add_argument("--symmetry_weight", type=float, default=0.01, help="对称损失权重 (default: 0.01)")
    parser.add_argument("--mid_lr", type=float, default=3e-4, help="线性下探阶段的目标学习率")
    parser.add_argument(
        "--mid_lr_epoch",
        type=int,
        default=60,
        help="线性下探结束并切换到cosine的epoch",
    )
    # mixup/cutmix 与 EMA 配置
    parser.add_argument("--use_mixup", action="store_true", help="启用 Mixup/CutMix")
    parser.add_argument("--mixup_alpha", type=float, default=0.2, help="Mixup alpha")
    parser.add_argument("--cutmix_alpha", type=float, default=1.0, help="CutMix alpha")
    parser.add_argument("--mixup_prob", type=float, default=0.5, help="Mixup/CutMix概率")
    parser.add_argument(
        "--mixup_stop_epoch",
        type=int,
        default=100,
        help="达到该epoch后关闭Mixup/CutMix",
    )
    parser.add_argument("--use_ema", action="store_true", help="启用 EMA 评估")
    parser.add_argument("--ema_decay", type=float, default=0.9998, help="EMA 衰减系数")
    parser.add_argument(
        "--ema_start_epoch",
        type=int,
        default=0,
        help="EMA 开始更新的epoch (default: 0, 立即启用)",
    )
    parser.add_argument("--patience", type=int, default=30, help="早停patience (default: 30)")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="预热轮数 (default: 10)")
    parser.add_argument(
        "--plateau_epochs",
        type=int,
        default=0,
        help="warmup结束后保持基础学习率的轮数 (default: 0)",
    )
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载器工作进程数 (default: 4)")
    # 添加梯度裁剪和学习率调度
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪阈值 (default: 1.0)")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="梯度累积步数 (default: 1)",
    )
    parser.add_argument(
        "--lr_reduce_factor",
        type=float,
        default=0.6,
        help="验证集停滞时学习率缩放因子",
    )
    parser.add_argument(
        "--lr_reduce_patience",
        type=int,
        default=5,
        help="验证集停滞多少个epoch后缩放学习率",
    )
    parser.add_argument(
        "--save_dir",
        required=True,
        type=str,
        help="模型保存目录",
    )
    parser.add_argument(
        "--log_dir",
        required=True,
        type=str,
        help="训练日志输出目录",
    )
    parser.add_argument("--use_tensorboard", action="store_true", help="使用TensorBoard记录训练过程")
    parser.add_argument(
        "--use_strong_augmentation",
        action="store_true",
        help="使用强数据增强 (default: False, 使用保守增强防过拟合)",
    )
    parser.add_argument(
        "--repeated_aug",
        type=int,
        default=1,
        help="Repeated Augmentation重复次数 (DeiT技巧, default: 1=不重复, 推荐3)",
    )
    parser.add_argument(
        "--drop_path_rate",
        type=float,
        default=0.1,
        help="Stochastic Depth的drop path rate (default: 0.1, DeiT III推荐0.1-0.2)",
    )
    parser.add_argument("--resume", type=str, default=None, help="从检查点恢复训练")
    parser.add_argument(
        "--resume_reset_epochs",
        action="store_true",
        help="从检查点恢复时将epoch计数重置为0，按新的 --epochs 作为微调轮数",
    )
    parser.add_argument(
        "--resume_reset_optimizer",
        action="store_true",
        help="从检查点恢复时不加载优化器状态，使用当前超参重新初始化优化器",
    )
    parser.add_argument(
        "--resume_reset_best",
        action="store_true",
        help="从检查点恢复时重置最佳验证准确率 (best_acc) 为0",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="使用混合精度训练 (默认关闭，启用需 GPU 支持)",
    )
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="bf16",
        choices=["fp16", "bf16"],
        help="混合精度数据类型 (fp16 或 bf16，默认 bf16)",
    )
    parser.add_argument(
        "--use_compile",
        action="store_true",
        help="使用 torch.compile 加速模型 (PyTorch 2.0+, 默认关闭)",
    )
    parser.add_argument(
        "--cudnn_benchmark",
        action="store_true",
        help="启用 cudnn.benchmark 加速 (输入尺寸固定时效果显著)",
    )
    # 单卡提速措施
    parser.add_argument(
        "--use_tf32",
        action="store_true",
        help="启用 TF32 加速 (Ampere GPU: A100/3090/4090, ~10-20%%提速)",
    )
    parser.add_argument(
        "--channels_last",
        action="store_true",
        help="启用 Channels Last 内存格式 (对CNN有效，~10-30%%提速)",
    )
    parser.add_argument(
        "--fused_optimizer",
        action="store_true",
        help="启用 Fused AdamW 优化器 (PyTorch 2.0+, ~5-15%%提速)",
    )
    # TTA (Test-Time Augmentation) 配置
    parser.add_argument(
        "--use_tta",
        action="store_true",
        help="验证时使用TTA (Test-Time Augmentation)，通常提升0.5-2%%准确率",
    )
    parser.add_argument(
        "--tta_level",
        type=str,
        default="light",
        choices=["light", "medium", "heavy"],
        help="TTA强度: light(2x快速), medium(4x), heavy(8x最准但慢)",
    )
    # SWA (Stochastic Weight Averaging) 配置
    parser.add_argument(
        "--use_swa",
        action="store_true",
        help="启用SWA (Stochastic Weight Averaging)，训练后期平均权重提升泛化",
    )
    parser.add_argument(
        "--swa_start_epoch",
        type=int,
        default=None,
        help="SWA开始epoch (default: 总epochs的75%%)",
    )
    parser.add_argument(
        "--swa_freq",
        type=int,
        default=1,
        help="SWA更新频率，每多少个epoch更新一次 (default: 1)",
    )
    # DDP 配置
    parser.add_argument(
        "--use_ddp",
        action="store_true",
        help="启用 DDP 多卡训练 (使用 torchrun 或 torch.distributed.launch 启动)",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="DDP local rank (由 torch.distributed.launch 自动传入)",
    )

    args = parser.parse_args()
    
    #timestamp for output files
    _timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.save_dir = f"{args.save_dir}_{_timestamp}"
    args.log_dir = f"{args.log_dir}_{_timestamp}"

    # DDP 初始化
    if args.use_ddp:
        # 从环境变量获取 rank 和 world_size（torchrun 会自动设置）
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))

        if local_rank == -1:
            raise ValueError("DDP 模式需要使用 torchrun 或 torch.distributed.launch 启动")

        setup_ddp(local_rank, world_size)
        device = torch.device(f"cuda:{local_rank}")

        # 只在主进程打印信息
        if is_main_process():
            logger.info(f"启用 DDP 训练: world_size={world_size}, rank={rank}, local_rank={local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0

    job_id = os.environ.get("SLURM_JOB_ID") or ""
    job_name = os.environ.get("SLURM_JOB_NAME")
    if not job_name:
        job_name = f"vision_mlbn_{args.model}"

    # 只在主进程配置日志
    if is_main_process():
        stdout_path, stderr_path = configure_logging(args.log_dir, job_name=job_name, job_id=job_id)
        if stdout_path:
            logger.info(f"标准输出重定向到: {stdout_path}")
        if stderr_path:
            logger.info(f"标准错误重定向到: {stderr_path}")

    seed = args.seed
    if seed is not None and seed < 0:
        seed = None

    # DDP 模式下，每个进程使用不同的随机种子
    if args.use_ddp and seed is not None:
        seed = seed + dist.get_rank()

    set_seed(seed)

    if is_main_process():
        logger.info(f"使用设备: {device}")

    # 创建保存目录（只在主进程）
    if is_main_process():
        os.makedirs(args.save_dir, exist_ok=True)

    # 设置TensorBoard（只在主进程）
    writer = None
    if args.use_tensorboard and is_main_process():
        if not args.log_dir:
            raise ValueError("使用TensorBoard时必须提供有效的日志目录 --log_dir")
        tensorboard_dir = os.path.join(args.log_dir, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(tensorboard_dir)
        logger.info(f"TensorBoard日志保存在: {tensorboard_dir}")

    # 加载数据
    train_loader, val_loader, num_classes = get_tiny_imagenet_data_loaders(
        args.dataset_path,
        args.batch_size,
        args.num_workers,
        args.use_strong_augmentation,
        seed=seed,
        repeated_aug=args.repeated_aug,
        use_ddp=args.use_ddp,
        input_size=args.input_size,
    )

    # 创建模型 (支持DeiT风格的Stochastic Depth)
    if is_main_process():
        logger.info(f"创建模型: {args.model}")
    model = create_model(args.model, num_classes, img_size=args.input_size, drop_path_rate=args.drop_path_rate)
    model = model.to(device)

    # DDP: 用 DistributedDataParallel 包装模型
    if args.use_ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,  # MLBN 模型有部分参数可能不参与某些批次的梯度计算
            bucket_cap_mb=25,  # 增大bucket减少通信次数
            gradient_as_bucket_view=True,  # 减少内存拷贝
        )

        # 通信优化：FP16梯度压缩（减少50%通信数据量）
        try:
            from torch.distributed.algorithms.ddp_comm_hooks import (
                default_hooks as default,
            )

            model.register_comm_hook(None, default.fp16_compress_hook)
            if is_main_process():
                logger.info("已启用FP16梯度压缩，通信数据量减少50%")
        except ImportError:
            if is_main_process():
                logger.warning("当前PyTorch版本不支持通信压缩hook，跳过")

        if is_main_process():
            logger.info(
                f"模型已用 DDP 包装 (device_ids=[{local_rank}], find_unused_parameters=True, "
                f"bucket_cap_mb=25, gradient_as_bucket_view=True)"
            )

    # 使用 torch.compile 加速模型 (PyTorch 2.0+)
    if hasattr(torch, "compile") and args.use_compile:
        logger.info("启用 torch.compile 加速模型...")
        model = torch.compile(model, mode="reduce-overhead")
        logger.info("torch.compile 编译完成")

    # 启用 cudnn benchmark 加速（输入尺寸固定时效果显著）
    if device.type == "cuda" and args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        logger.info("启用 cudnn.benchmark 加速")

    # 启用 TF32 模式 (Ampere GPU: A100/3090/4090, ~10-20%提速，精度损失可忽略)
    if device.type == "cuda" and args.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("启用 TF32 加速 (Ampere+ GPU)")

    # 启用 Channels Last 内存格式 (对CNN有效，~10-30%提速)
    if device.type == "cuda" and args.channels_last:
        model = model.to(memory_format=torch.channels_last)  # type: ignore
        logger.info("启用 Channels Last 内存格式")

    use_amp = False
    amp_dtype = None
    if args.use_amp:
        if device.type != "cuda":
            logger.warning("检测到 --use_amp 但当前设备不是 CUDA，自动关闭混合精度。")
        else:
            use_amp = True
            if args.amp_dtype == "bf16":
                supports_bf16 = torch.cuda.is_bf16_supported() if hasattr(torch.cuda, "is_bf16_supported") else False
                if supports_bf16:
                    amp_dtype = torch.bfloat16
                else:
                    logger.warning("GPU 不支持 bfloat16，将使用 float16 进行混合精度训练。")
                    amp_dtype = torch.float16
            else:
                amp_dtype = torch.float16
            dtype_name = "bf16" if amp_dtype == torch.bfloat16 else "fp16"
            logger.info(f"启用混合精度训练 (dtype={dtype_name})。")
    else:
        logger.info("未启用混合精度训练。")

    # 组装损失与增广（训练与验证分开）
    val_criterion = nn.CrossEntropyLoss()
    train_criterion_hard = SymmetryLossWrapper(
        LabelSmoothingCrossEntropy(smoothing=args.label_smoothing), symmetry_weight=args.symmetry_weight
    )
    train_criterion_soft = None
    mixup_fn = None
    if args.use_mixup:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            prob=args.mixup_prob,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=0.05,
            num_classes=num_classes,
        )
        train_criterion_soft = SymmetryLossWrapper(SoftTargetCrossEntropy(), symmetry_weight=args.symmetry_weight)

    # 优化器改为 AdamW
    # fused=True: 使用融合实现，减少kernel启动开销 (~5-15%提速, PyTorch 2.0+)
    use_fused_optimizer = args.fused_optimizer and device.type == "cuda"
    if use_fused_optimizer:
        # 检查PyTorch版本是否支持fused参数
        try:
            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.lr,
                betas=(0.9, 0.999),
                weight_decay=args.weight_decay,
                fused=True,
            )
            logger.info("启用 Fused AdamW 优化器")
        except TypeError:
            # 旧版PyTorch不支持fused参数
            logger.warning("当前PyTorch版本不支持fused优化器，使用标准AdamW")
            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.lr,
                betas=(0.9, 0.999),
                weight_decay=args.weight_decay,
            )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
        )
    scaler = GradScaler(enabled=use_amp)

    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_reduce_factor,
        patience=args.lr_reduce_patience,
        threshold=1e-4,
        threshold_mode="rel",
        min_lr=args.min_lr,
    )

    # EMA（可选）
    model_ema = None
    ema_active = False
    if args.use_ema and args.ema_start_epoch <= 0:
        model_ema = ModelEmaV2(model, decay=args.ema_decay)
        ema_active = True

    # 训练循环
    best_acc = 0
    best_epoch = 0  # 跟踪最佳验证准确率的epoch
    start_epoch = 0
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001)
    effective_base_lr = args.lr  # 有效基础学习率（用于cosine+warmup）

    # 动态正则化增强相关变量
    reg_enhancement_triggered = False
    original_mixup_alpha = args.mixup_alpha if mixup_fn is not None else None
    original_mixup_prob = args.mixup_prob if mixup_fn is not None else None
    original_weight_decay = args.weight_decay

    if args.resume:
        logger.info(f"从检查点恢复: {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

        if args.resume_reset_optimizer:
            logger.info("重置优化器状态，不加载检查点中的优化器参数")
        else:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint.get("epoch", 0)
        best_acc = checkpoint.get("best_acc", best_acc)

        if args.resume_reset_epochs:
            logger.info("重置epoch计数为0（微调模式）")
            start_epoch = 0

        if args.resume_reset_best:
            logger.info("重置最佳验证准确率为0")
            best_acc = 0.0
        else:
            logger.info(f"恢复最佳验证准确率: {best_acc:.2f}%")

        if not args.resume_reset_epochs and args.epochs <= start_epoch:
            new_total_epochs = start_epoch + args.epochs
            logger.warning(
                "传入的 --epochs (%d) 小于或等于检查点epoch (%d)， 将epochs解释为附加训练轮数，总训练轮数调整为 %d",
                args.epochs,
                start_epoch,
                new_total_epochs,
            )
            args.epochs = new_total_epochs

    if is_main_process():
        logger.info(f"开始训练，共 {args.epochs} 个epoch")
        logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 初始化SWA
    swa_model = None
    swa_active = False
    if args.use_swa:
        swa_start = args.swa_start_epoch if args.swa_start_epoch is not None else int(args.epochs * 0.75)
        logger.info(f"SWA 将在 Epoch {swa_start} 启用 (更新频率: 每{args.swa_freq}个epoch)")

    # TTA配置
    if args.use_tta:
        tta_transforms = get_tta_transforms(args.tta_level)
        logger.info(f"TTA 已启用 (level={args.tta_level}, {len(tta_transforms)}x增强)")
    else:
        tta_transforms = None

    mixup_disabled_logged = False

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.perf_counter()
        # 设置Repeated Augmentation sampler的epoch（用于随机化）
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        # 调整学习率（基于有效基础学习率，而不是原始学习率）
        current_lr = adjust_learning_rate_cosine(
            optimizer,
            epoch,
            effective_base_lr,
            args.warmup_epochs,
            args.plateau_epochs,
            args.epochs,
            args.min_lr,
            args.mid_lr,
            args.mid_lr_epoch,
        )

        # 训练
        if args.use_ema and not ema_active and epoch >= args.ema_start_epoch:
            model_ema = ModelEmaV2(model, decay=args.ema_decay)
            ema_active = True
            logger.info(f"Epoch {epoch}: 启用EMA (decay={args.ema_decay}, start_epoch={args.ema_start_epoch})")

        # 动态正则化：如果检测到过拟合，即使过了mixup_stop_epoch也继续使用Mixup
        # （reg_enhancement_triggered在之前的epoch中可能被设置）
        if mixup_fn is not None and epoch >= args.mixup_stop_epoch and not reg_enhancement_triggered:
            if not mixup_disabled_logged:
                logger.info(f"Epoch {epoch}: 停用Mixup/CutMix以集中拟合干净标签")
                mixup_disabled_logged = True
            active_mixup_fn = None
        else:
            active_mixup_fn = mixup_fn
            # 如果因为过拟合重新启用，记录日志
            if reg_enhancement_triggered and epoch >= args.mixup_stop_epoch and mixup_disabled_logged:
                logger.info(f"Epoch {epoch}: 因过拟合重新启用Mixup/CutMix")
                mixup_disabled_logged = False  # 重置标志以便后续再次记录

        if active_mixup_fn is not None and train_criterion_soft is not None:
            active_train_criterion = train_criterion_soft
        else:
            active_train_criterion = train_criterion_hard

        train_loss, train_acc1, train_acc5 = train_epoch(
            model,
            train_loader,
            active_train_criterion,
            optimizer,
            active_mixup_fn,
            model_ema if ema_active else None,
            device,
            epoch,
            args.max_grad_norm,
            args.gradient_accumulation_steps,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            scaler=scaler,
        )

        # 计算并记录对称损失
        symmetry_loss = None
        if isinstance(active_train_criterion, SymmetryLossWrapper) and active_train_criterion.symmetry_weight > 0.0:
            try:
                symmetry_loss = float(model.encoder.compute_weighted_symmetry_loss())
            except Exception as e:
                if is_main_process():
                    logger.warning(f"计算对称损失失败: {e}")

        # SWA更新（在训练后期）
        if args.use_swa:
            swa_start = args.swa_start_epoch if args.swa_start_epoch is not None else int(args.epochs * 0.75)
            if epoch >= swa_start and (epoch - swa_start) % args.swa_freq == 0:
                if swa_model is None:
                    swa_model = SWAModel(model, device)
                    if is_main_process():
                        logger.info(f"Epoch {epoch}: 启用SWA (start_epoch={swa_start})")
                swa_model.update(model)

        # 验证
        eval_model = model_ema.module if (ema_active and model_ema is not None) else model

        # 使用TTA或普通验证（带 DataLoader 错误恢复）
        try:
            if args.use_tta and tta_transforms is not None:
                val_loss, val_acc1, val_acc5 = validate_with_tta(
                    eval_model,
                    val_loader,
                    val_criterion,
                    device,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                    tta_transforms=tta_transforms,
                )
            else:
                val_loss, val_acc1, val_acc5 = validate(
                    eval_model,
                    val_loader,
                    val_criterion,
                    device,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                )
        except RuntimeError as e:
            if "DataLoader worker" in str(e) or "Segmentation fault" in str(e):
                logger.warning(f"DataLoader 错误: {e}")
                logger.warning("重新创建验证 DataLoader (num_workers=0)...")

                # 重新创建验证 DataLoader，禁用多进程
                from torch.utils.data import DataLoader

                val_dataset = val_loader.dataset
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=0,  # 禁用多进程
                    pin_memory=True,
                )

                # 重试验证
                if args.use_tta and tta_transforms is not None:
                    val_loss, val_acc1, val_acc5 = validate_with_tta(
                        eval_model,
                        val_loader,
                        val_criterion,
                        device,
                        use_amp=use_amp,
                        amp_dtype=amp_dtype,
                        tta_transforms=tta_transforms,
                    )
                else:
                    val_loss, val_acc1, val_acc5 = validate(
                        eval_model,
                        val_loader,
                        val_criterion,
                        device,
                        use_amp=use_amp,
                        amp_dtype=amp_dtype,
                    )
                logger.info("验证成功恢复")
            else:
                raise

        # 验证后清理，确保释放临时tensor
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        prev_lr = optimizer.param_groups[0]["lr"]
        lr_scheduler.step(val_acc1)
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr < prev_lr - 1e-12:
            scale = current_lr / max(prev_lr, 1e-12)
            effective_base_lr = max(args.min_lr, effective_base_lr * scale)
            logger.info(f"ReduceLROnPlateau: 学习率从 {prev_lr:.6f} 调整为 {current_lr:.6f}")

        # 记录到TensorBoard
        if writer:
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Validation", val_loss, epoch)
            if symmetry_loss is not None:
                writer.add_scalar("Loss/Symmetry", symmetry_loss, epoch)
            writer.add_scalar("Accuracy/Train_Acc1", train_acc1, epoch)
            writer.add_scalar("Accuracy/Val_Acc1", val_acc1, epoch)
            writer.add_scalar("Accuracy/Train_Acc5", train_acc5, epoch)
            writer.add_scalar("Accuracy/Val_Acc5", val_acc5, epoch)
            writer.add_scalar("Learning_Rate", current_lr, epoch)
            # 每10个epoch刷新一次，避免内存累积
            if (epoch + 1) % 10 == 0:
                writer.flush()

        # 每轮训练输出统计信息（只在主进程）
        if is_main_process():
            symmetry_info = f"Symmetry Loss={symmetry_loss:.6f}, " if symmetry_loss is not None else ""
            logger.info(
                f"Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Train Acc@1={train_acc1:.2f}%, "
                f"Val Loss={val_loss:.4f}, Val Acc@1={val_acc1:.2f}%, "
                f"{symmetry_info}LR={current_lr:.6f}"
            )

        # 保存最佳模型（只在主进程）
        if val_acc1 > best_acc:
            best_acc = val_acc1
            best_epoch = epoch  # 更新最佳epoch
            if is_main_process():
                best_model_path = os.path.join(args.save_dir, "best_model.pth")
                # DDP 模式下需要获取模型的 module
                model_to_save = model.module if args.use_ddp else model
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": (
                            model_ema.module.state_dict() if model_ema is not None else model_to_save.state_dict()
                        ),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_acc": best_acc,
                        "args": args,
                    },
                    best_model_path,
                )
                logger.info(f"验证提升，保存最佳模型到: {best_model_path} (Acc1={best_acc:.2f}%)")

        # 动态正则化增强：检测到过拟合时自动增强正则化
        epochs_since_best = epoch - best_epoch
        overfitting_threshold = 20  # 验证停滞20个epoch后触发
        val_degradation = best_acc - val_acc1  # 当前验证准确率相对最佳的下降幅度

        if epochs_since_best >= overfitting_threshold and val_degradation > 0.2:
            if not reg_enhancement_triggered:
                reg_enhancement_triggered = True
                logger.warning(
                    f"检测到过拟合迹象 (最佳Epoch {best_epoch}: {best_acc:.2f}%, "
                    f"当前: {val_acc1:.2f}%, 停滞{epochs_since_best}个epoch)，启用动态正则化增强"
                )

            # 策略1：如果Mixup已停用，重新启用它（过拟合时需要更强正则化）
            if mixup_fn is not None and active_mixup_fn is None and epoch >= args.mixup_stop_epoch:
                # 重新启用Mixup并增强其强度
                new_alpha = min(original_mixup_alpha * 1.2, 0.5)  # type: ignore
                new_prob = min(original_mixup_prob * 1.1, 0.9)  # type: ignore

                # 直接设置属性（timm Mixup类使用 mixup_alpha 和 mix_prob）
                mixup_fn.mixup_alpha = new_alpha
                mixup_fn.mix_prob = new_prob

                logger.info(f"重新启用Mixup以对抗过拟合: alpha={new_alpha:.3f}, prob={new_prob:.3f}")
                # 更新active_mixup_fn（在下一个epoch会生效）

            # 策略2：如果Mixup还在使用，增强其强度
            elif mixup_fn is not None and active_mixup_fn is not None and original_mixup_alpha is not None:
                # 逐步增强Mixup alpha（最多增强到0.5）
                # timm Mixup类使用 mixup_alpha 和 mix_prob 属性
                old_alpha = mixup_fn.mixup_alpha
                old_prob = mixup_fn.mix_prob

                new_alpha = min(old_alpha * 1.05, 0.5)
                new_prob = min(old_prob * 1.02, 0.9)

                if new_alpha > old_alpha:
                    mixup_fn.mixup_alpha = new_alpha
                    logger.info(f"增强Mixup alpha: {new_alpha:.3f} (原始{original_mixup_alpha:.3f})")

                if new_prob > old_prob:
                    mixup_fn.mix_prob = new_prob
                    logger.info(f"增强Mixup prob: {new_prob:.3f} (原始{original_mixup_prob:.3f})")

            # 增强weight_decay（每5个epoch增强一次，避免过于频繁）
            if epochs_since_best % 5 == 0:
                for param_group in optimizer.param_groups:
                    old_wd = param_group["weight_decay"]
                    new_wd = min(old_wd * 1.05, original_weight_decay * 1.5)  # 最多增强50%
                    if new_wd > old_wd:
                        param_group["weight_decay"] = new_wd
                        logger.info(f"增强权重衰减: {old_wd:.6f} -> {new_wd:.6f} (原始{original_weight_decay:.6f})")

            # 记录过拟合指标
            train_val_gap = train_acc1 - val_acc1
            if writer:
                writer.add_scalar("Overfitting/Train_Val_Gap", train_val_gap, epoch)
                writer.add_scalar("Overfitting/Epochs_Since_Best", epochs_since_best, epoch)
                writer.add_scalar("Overfitting/Val_Degradation", val_degradation, epoch)

        # 早停检查
        should_stop, _ = early_stopping(val_acc1)

        if should_stop:
            logger.info("")
            logger.info(f"早停触发！最佳验证准确率: {best_acc:.2f}%")
            logger.info(f"连续 {early_stopping.counter} 个epoch验证准确率未提升")
            logger.info(f"早停配置: patience={early_stopping.patience}, min_delta={early_stopping.min_delta}")
            break

        # 定期保存检查点（只在主进程）
        if (epoch + 1) % 20 == 0 and is_main_process():
            checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            # DDP 模式下需要获取模型的 module
            model_to_save = model.module if args.use_ddp else model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "args": args,
                },
                checkpoint_path,
            )
            logger.info(f"保存检查点: {checkpoint_path}")

        # 周期性清理CUDA缓存，避免碎片化累积
        if (epoch + 1) % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            if is_main_process():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"CUDA内存清理完成 | 已分配: {allocated:.2f}GB | 已保留: {reserved:.2f}GB")

        # 简洁的输出格式（只在主进程）
        if is_main_process():
            logger.info(
                f"Epoch {epoch + 1:2d}/{args.epochs} | "
                f"训练: {train_acc1:.1f}% | 验证: {val_acc1:.1f}% | 最佳: {best_acc:.1f}% | "
                f"LR: {current_lr:.6f} | "
                f"耗时: {time.perf_counter() - epoch_start_time:.2f}s"
            )

    logger.info(f"训练完成！最佳验证准确率: {best_acc:.2f}%")

    # SWA最终评估
    if args.use_swa and swa_model is not None and swa_model.n_averaged > 0:
        if is_main_process():
            logger.info(f"正在完成SWA (共平均了 {swa_model.n_averaged} 个模型)...")
        swa_model.finalize(train_loader, device, use_amp=use_amp, amp_dtype=amp_dtype)

        # 使用SWA模型进行最终验证
        if args.use_tta and tta_transforms is not None:
            swa_loss, swa_acc1, swa_acc5 = validate_with_tta(
                swa_model.model,
                val_loader,
                val_criterion,
                device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                tta_transforms=tta_transforms,
            )
        else:
            swa_loss, swa_acc1, swa_acc5 = validate(
                swa_model.model,
                val_loader,
                val_criterion,
                device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
        logger.info(f"SWA模型验证准确率: {swa_acc1:.2f}% (vs 最佳EMA/普通: {best_acc:.2f}%)")

        # 如果SWA更好，保存SWA模型（只在主进程）
        if swa_acc1 > best_acc and is_main_process():
            swa_path = os.path.join(args.save_dir, "swa_model.pth")
            torch.save(swa_model.model.state_dict(), swa_path)
            logger.info(f"SWA模型更优！已保存到: {swa_path}")

    if writer:
        writer.close()

    # 清理 DDP
    if args.use_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()
