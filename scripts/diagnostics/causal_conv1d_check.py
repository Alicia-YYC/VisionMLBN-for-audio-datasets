"""Checking if causal-conv1d is installed properly. This script should be run on a computing node."""

import torch
import torch.nn.functional as F
from causal_conv1d import causal_conv1d_fn

device = "cuda"
batch, dim, seqlen, width = 2, 64, 128, 4
x = torch.randn(batch, dim, seqlen, device=device, dtype=torch.float16)
weight = torch.randn(dim, width, device=device, dtype=torch.float16)
bias = torch.randn(dim, device=device, dtype=torch.float16)

# 1. Fast CUDA kernel
out_fast = causal_conv1d_fn(x, weight, bias, activation="silu")

# 2. Reference PyTorch implementation
x_padded = F.pad(x, (width - 1, 0))
out_ref = F.conv1d(x_padded, weight.unsqueeze(1), bias, groups=dim)
out_ref = F.silu(out_ref)

# Compare
diff = (out_fast - out_ref).abs().max().item()
print(f"Max difference: {diff:.6f}")
if diff < 1e-2:
    print("Verification Successful: CUDA kernels match PyTorch reference!")
else:
    print("Verification Failed: Numerical mismatch.")
