"""
VisionMLBN 性能分析脚本

使用 PyTorch Profiler 分析模型瓶颈
"""

import argparse
import os
import sys

import torch

# 禁用 torch.compile (dynamo) - 与 Mamba SSM 的 Triton 内核不兼容
# 必须在导入模型之前设置
import torch._dynamo
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.profiler import ProfilerActivity, profile, record_function, schedule

torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from VisionMLBN import vim_tiny_patch16_224_mlbn


def create_dummy_data(batch_size, device):
    """创建虚拟数据用于性能分析"""
    data = torch.randn(batch_size, 3, 224, 224, device=device)
    target = torch.randint(0, 200, (batch_size,), device=device)
    return data, target


def warmup(model, data, num_iterations=10):
    """预热 GPU，确保 CUDA 内核已编译"""
    print(f"预热 {num_iterations} 次迭代...")
    model.eval()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(data)
    torch.cuda.synchronize()
    print("预热完成")


def profile_inference(model, data, args):
    """分析推理性能"""
    print("\n" + "=" * 60)
    print("推理性能分析")
    print("=" * 60)

    model.eval()

    # 创建 profiler (with_stack 会导致 trace 文件过大，默认关闭)
    with (
        profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=args.export_stacks,
            with_flops=True,
        ) as prof,
        torch.no_grad(),
    ):
        for _ in range(args.num_iterations):
            with record_function("model_inference"):
                if args.use_amp:
                    with autocast(dtype=torch.bfloat16):
                        _ = model(data)
                else:
                    _ = model(data)
            torch.cuda.synchronize()

    # 打印按 CUDA 时间排序的结果
    print("\n📊 按 CUDA 时间排序 (Top 20):")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    # 打印按 CPU 时间排序的结果
    print("\n📊 按 CPU 时间排序 (Top 20):")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    # 打印按显存使用排序的结果
    print("\n📊 按显存使用排序 (Top 20):")
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))

    # 导出 Chrome trace (可在 chrome://tracing 查看)
    if not args.no_trace:
        trace_path = os.path.join(args.output_dir, "inference_trace.json")
        prof.export_chrome_trace(trace_path)
        print(f"\n✅ Chrome trace 已导出到: {trace_path}")
    else:
        print("\n⏭️  跳过 Chrome trace 导出 (--no_trace)")

    # 导出堆栈信息
    if args.export_stacks:
        stacks_path = os.path.join(args.output_dir, "inference_stacks.txt")
        prof.export_stacks(stacks_path, "self_cuda_time_total")
        print(f"✅ 堆栈信息已导出到: {stacks_path}")

    return prof


def profile_training_step(model, data, target, criterion, optimizer, scaler, args):
    """分析单步训练性能"""
    print("\n" + "=" * 60)
    print("训练步骤性能分析")
    print("=" * 60)

    model.train()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=args.export_stacks,
        with_flops=True,
    ) as prof:
        for _ in range(args.num_iterations):
            optimizer.zero_grad()

            with record_function("forward"):
                if args.use_amp:
                    with autocast(dtype=torch.bfloat16):
                        output = model(data)
                        loss = criterion(output, target)
                else:
                    output = model(data)
                    loss = criterion(output, target)

            with record_function("backward"):
                if args.use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            with record_function("optimizer_step"):
                if args.use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

            torch.cuda.synchronize()

    # 打印结果
    print("\n📊 训练步骤 - 按 CUDA 时间排序 (Top 25):")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))

    print("\n📊 训练步骤 - 按操作分组:")
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=15))

    # 导出 trace
    if not args.no_trace:
        trace_path = os.path.join(args.output_dir, "training_trace.json")
        prof.export_chrome_trace(trace_path)
        print(f"\n✅ Chrome trace 已导出到: {trace_path}")
    else:
        print("\n⏭️  跳过 Chrome trace 导出 (--no_trace)")

    return prof


def profile_with_schedule(model, data, target, criterion, optimizer, scaler, args):
    """使用调度器进行更详细的分析"""
    print("\n" + "=" * 60)
    print("详细训练分析 (带调度器)")
    print("=" * 60)

    model.train()

    # 调度器: skip 2 iterations, warmup 2, active 6, repeat 1
    prof_schedule = schedule(wait=2, warmup=2, active=6, repeat=1)

    trace_path = os.path.join(args.output_dir, "scheduled_trace")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_path),
        record_shapes=True,
        profile_memory=True,
        with_stack=args.export_stacks,
    ) as prof:
        for _ in range(12):  # 2 + 2 + 6 + 2 = 12
            optimizer.zero_grad()

            if args.use_amp:
                with autocast(dtype=torch.bfloat16):
                    output = model(data)
                    loss = criterion(output, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            torch.cuda.synchronize()
            prof.step()

    print(f"\n✅ TensorBoard trace 已导出到: {trace_path}")
    print("   使用命令查看: tensorboard --logdir=" + trace_path)

    return prof


def analyze_model_layers(model, data, args):
    """分析各层的时间占比"""
    print("\n" + "=" * 60)
    print("模型各层时间分析")
    print("=" * 60)

    model.eval()

    # 为各主要模块添加 hooks
    layer_times = {}

    def make_hook(name):
        def hook(module, input, output):
            torch.cuda.synchronize()

        return hook

    # 注册 hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.LayerNorm, nn.MultiheadAttention)):
            hooks.append(module.register_forward_hook(make_hook(name)))

    # 使用 profiler 分析各层
    with (
        profile(
            activities=[ProfilerActivity.CUDA],
            record_shapes=True,
            with_modules=True,
        ) as prof,
        torch.no_grad(),
    ):
        for _ in range(args.num_iterations):
            if args.use_amp:
                with autocast(dtype=torch.bfloat16):
                    _ = model(data)
            else:
                _ = model(data)
            torch.cuda.synchronize()

    # 移除 hooks
    for h in hooks:
        h.remove()

    # 按模块分组打印
    print("\n📊 按模块分组的时间统计:")
    print(prof.key_averages(group_by_input_shape=False).table(sort_by="cuda_time_total", row_limit=30))


def print_summary(args, model):
    """打印分析摘要"""
    print("\n" + "=" * 60)
    print("性能分析配置摘要")
    print("=" * 60)

    # 模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"模型参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数: {trainable_params / 1e6:.2f}M")
    print(f"Batch Size: {args.batch_size}")
    print("输入尺寸: (3, 224, 224)")
    print(f"使用 AMP: {args.use_amp}")
    print(f"分析迭代次数: {args.num_iterations}")

    # GPU 信息
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name()}")
        print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"当前显存占用: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"显存缓存: {torch.cuda.memory_reserved() / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="VisionMLBN 性能分析")
    parser.add_argument("--batch_size", type=int, default=32, help="分析用的 batch size (默认: 32)")
    parser.add_argument("--num_iterations", type=int, default=20, help="分析迭代次数 (默认: 20)")
    parser.add_argument("--use_amp", action="store_true", help="使用混合精度 (默认: 关闭)")
    parser.add_argument("--use_compile", action="store_true", help="使用 torch.compile (默认: 关闭)")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./profile_output",
        help="分析结果输出目录",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["inference", "training", "scheduled", "layers", "all"],
        help="分析模式",
    )
    parser.add_argument("--export_stacks", action="store_true", help="导出堆栈信息")
    parser.add_argument(
        "--no_trace",
        action="store_true",
        help="不导出 Chrome trace 文件 (可减少磁盘占用)",
    )

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    if device.type != "cuda":
        print("⚠️  警告: 未检测到 CUDA，性能分析结果可能不准确")

    # 创建模型
    print("创建模型...")
    model = vim_tiny_patch16_224_mlbn(num_classes=200, drop_path_rate=0.2)
    model = model.to(device)

    # torch.compile - 跳过，因为与 Mamba SSM 的 Triton 内核不兼容
    if args.use_compile:
        print("⚠️  跳过 torch.compile (与 Mamba SSM Triton 内核不兼容)")
        print("   Profiler 将分析未编译的模型性能")

    # 创建虚拟数据
    data, target = create_dummy_data(args.batch_size, device)

    # 打印摘要
    print_summary(args, model)

    # 预热
    warmup(model, data)

    # 创建优化器和损失函数 (用于训练分析)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    # 运行分析
    if args.mode in ["inference", "all"]:
        profile_inference(model, data, args)

    if args.mode in ["training", "all"]:
        profile_training_step(model, data, target, criterion, optimizer, scaler, args)

    if args.mode in ["scheduled", "all"]:
        # 重新创建优化器 (因为之前的分析可能改变了状态)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
        profile_with_schedule(model, data, target, criterion, optimizer, scaler, args)

    if args.mode in ["layers", "all"]:
        analyze_model_layers(model, data, args)

    print("\n" + "=" * 60)
    print("✅ 性能分析完成!")
    print("=" * 60)
    print(f"\n结果保存在: {args.output_dir}")
    print("\n查看 Chrome trace:")
    print("  1. 打开 Chrome 浏览器")
    print("  2. 访问 chrome://tracing")
    print(f"  3. 加载 {args.output_dir}/*.json 文件")


if __name__ == "__main__":
    main()
