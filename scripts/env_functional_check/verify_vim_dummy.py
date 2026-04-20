import sys

import torch
from vim.models_mamba import VisionMamba


def test_vim_installation():
    # 1. Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    if device == "cpu":
        print("WARNING: Vim is designed for CUDA. CPU execution might fail or be extremely slow.")
        # If running on a cluster, failing here prevents wasting cluster time
        sys.exit(1)

    # 2. Initialize a tiny version of the Vim model
    # We use standard ImageNet size (224x224)
    model = VisionMamba(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        final_pool_type="mean",
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        bimamba_type="v2",
        if_cls_token=True,
        if_devide_out=True,
        use_middle_cls_token=True,
        num_classes=1000,
    ).to(device)

    print("Model initialized successfully.")

    # 3. Create dummy input (Batch Size 1, 3 Channels, 224x224 Image)
    x = torch.randn(1, 3, 224, 224).to(device)

    # 4. Run Forward Pass
    try:
        with torch.no_grad():
            output = model(x)
        print(f"Forward pass successful! Output shape: {output.shape}")
        print("✅ Vim installation is CORRECT.")
    except Exception as e:
        print("❌ Forward pass FAILED.")
        print(f"Error: {e}")


if __name__ == "__main__":
    test_vim_installation()
