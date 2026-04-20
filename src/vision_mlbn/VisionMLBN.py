"""
Vision Mamba with MLBN Encoder
将Vision Mamba模型中的编码器替换为MLBN.py中的VisionMLBN_encoder
完全保持Vision Mamba的原始结构，只替换编码器部分
"""

import math
import random
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入MLBN编码器
from MLBN import VisionMLBN_encoder
from timm.models.layers import DropPath, lecun_normal_, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, _load_weights

# 导入rope相关模块
try:
    from rope import VisionRotaryEmbeddingFast
except ImportError:
    # 如果rope模块不可用，定义一个简单的替代
    class VisionRotaryEmbeddingFast(nn.Module):
        def __init__(self, dim, pt_seq_len=16, ft_seq_len=None, **kwargs):
            super().__init__()
            # 简单的身份映射作为替代
            pass

        def forward(self, x):
            return x


try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


__all__ = [
    "vim_tiny_patch16_224_mlbn",
    "vim_tiny_patch16_stride8_224_mlbn",
    "vim_small_patch16_224_mlbn",
    "vim_small_patch16_stride8_224_mlbn",
    "vim_base_patch16_224_mlbn",
]


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding

    标准模式输出: (2, B, num_patches, embed_dim)
        - 第一个分量 y1: 原图像的 PatchEmbed 输出
        - 第二个分量 y2: 中心对称图像的 PatchEmbed 输出，并在序列维度上翻转
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        stride=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size

        base_grid_h = (img_size[0] - patch_size[0]) // stride + 1
        base_grid_w = (img_size[1] - patch_size[1]) // stride + 1
        self.base_grid_size = (base_grid_h, base_grid_w)
        self.grid_size = (base_grid_h, base_grid_w)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """
        输入:
            x: (B, C, H, W) - 输入图像
        输出:
            out: (2, B, num_patches, embed_dim)
                - out[0]: 原图像的 PatchEmbed
                - out[1]: 中心对称图像的 PatchEmbed + 序列维度翻转
        """
        B, C, H, W = x.shape
        assert self.img_size[0] == H and self.img_size[1] == W, (
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        )

        # 优化: 批量处理原图和翻转图，减少kernel调用次数
        x_flip = torch.flip(x, dims=[2, 3])  # 中心对称操作
        x_combined = torch.cat([x, x_flip], dim=0)  # (2B, C, H, W)

        # 单次卷积处理两个batch
        y_combined = self.proj(x_combined)  # (2B, embed_dim, H', W')

        if self.flatten:
            y_combined = y_combined.flatten(2).transpose(1, 2)  # (2B, num_patches, embed_dim)
            y1, y2 = y_combined[:B], y_combined[B:]
            # 在 num_patches 维度上翻转 y2
            y2 = torch.flip(y2, dims=[1])
        else:
            y1, y2 = y_combined[:B], y_combined[B:]

        y1 = self.norm(y1)
        y2 = self.norm(y2)

        # 堆叠输出: (2, B, num_patches, embed_dim)
        out = torch.stack([y1, y2], dim=0)
        return out


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class VisionMambaMLBN(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        stride=16,
        depth=24,
        embed_dim=192,
        d_state=16,
        channels=3,
        num_classes=1000,
        ssm_cfg=None,
        drop_rate=0.0,
        drop_path_rate=0.1,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        initializer_cfg=None,
        fused_add_norm=True,
        residual_in_fp32=True,
        device=None,
        dtype=None,
        ft_seq_len=None,
        pt_hw_seq_len=14,
        if_bidirectional=False,
        final_pool_type="none",
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        flip_img_sequences_ratio=-1.0,
        if_bimamba=False,
        bimamba_type="v2",
        cls_strategy="first",
        if_divide_out=True,
        init_layer_scale=None,
        cls_rope_mode=None,
        # MLBN编码器参数（默认值，按Vision Mamba标准）
        # mlbn_L参数已移除，现在使用depth参数
        mlbn_kernel_size=9,
        mlbn_d_state=16,  # Vision Mamba标准配置
        mlbn_d_conv=4,  # Vision Mamba标准配置
        mlbn_expand=2,  # Vision Mamba标准配置
        mlbn_head_dim=16,  # Vision Mamba标准配置
        mlbn_dropout_rate1=0.08,
        mlbn_num_heads=3,  # 默认值，实际会被各模型覆盖
        mlbn_dropout_rate2=0.08,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs)
        super().__init__()
        if cls_strategy is None:
            strategy = "none"
        elif isinstance(cls_strategy, str):
            strategy = cls_strategy.lower()
        else:
            raise TypeError("cls_strategy must be str or None")

        valid_strategies = {"none", "first", "middle", "last", "double"}
        if strategy not in valid_strategies:
            raise ValueError(f"Unsupported cls_strategy: {cls_strategy}")

        self.cls_strategy = strategy
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.if_bidirectional = if_bidirectional
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        self.flip_img_sequences_ratio = flip_img_sequences_ratio
        self.if_cls_token = strategy != "none"
        self.use_double_cls_token = strategy == "double"
        self.use_middle_cls_token = strategy == "middle"
        self.cls_at_last = strategy == "last"
        self.num_tokens = 2 if self.use_double_cls_token else (1 if self.if_cls_token else 0)

        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            stride=stride,
            in_chans=channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        if self.use_double_cls_token:
            self.cls_token_head = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.cls_token_tail = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        elif self.if_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, self.embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)

        if if_rope:
            half_head_dim = embed_dim // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(dim=half_head_dim, pt_seq_len=pt_hw_seq_len, ft_seq_len=hw_seq_len)
        # 分类头 - 使用标准ViT设计
        if num_classes > 0:
            self.head = nn.Sequential(
                nn.LayerNorm(self.num_features),  # ViT标准：pre-classifier norm
                nn.Linear(self.num_features, num_classes),
            )
        else:
            self.head = nn.Identity()

        # Stochastic depth (not used in MLBN encoder, but kept for compatibility)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        def _normalize_cls_rope_mode(mode):
            if mode is None:
                return None
            if isinstance(mode, str):
                m = mode.lower()
                if m in {"none", "off"}:
                    return None
                if m in {"first", "cls", "head"}:
                    return "first"
                if m in {"last", "tail"}:
                    return "last"
                if m in {"middle", "medium", "mid"}:
                    return "medium"
                if m in {"-1"}:  # allow string -1
                    return -1
                raise ValueError(f"Unsupported cls_rope_mode: {mode}")
            if isinstance(mode, int):
                return mode
            raise TypeError("cls_rope_mode must be None, str, or int")

        strategy_default = {
            "none": None,
            "first": "first",
            "middle": "medium",
            "last": "last",
            "double": "first",
        }
        effective_cls_rope_mode = _normalize_cls_rope_mode(cls_rope_mode)
        if self.if_cls_token:
            if effective_cls_rope_mode is None:
                effective_cls_rope_mode = strategy_default[self.cls_strategy]
        else:
            effective_cls_rope_mode = None

        # 使用MLBN编码器替换原始的transformer blocks
        # MLBN编码器层数应该与Vision Mamba的depth匹配
        self.encoder = VisionMLBN_encoder(
            L=depth,  # 使用Vision Mamba的depth参数
            input_dim=embed_dim,
            kernel_size=mlbn_kernel_size,
            d_state=mlbn_d_state,
            d_conv=mlbn_d_conv,
            expand=mlbn_expand,
            head_dim=mlbn_head_dim,
            dropout_rate1=mlbn_dropout_rate1,
            num_heads=mlbn_num_heads,
            dropout_rate2=mlbn_dropout_rate2,
            cls_rope_mode=effective_cls_rope_mode,
        )
        self.cls_rope_mode = effective_cls_rope_mode

        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon, **factory_kwargs)

        # self.pre_logits = nn.Identity()

        # original init
        self.patch_embed.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=0.02)
        if self.if_cls_token:
            if self.use_double_cls_token:
                trunc_normal_(self.cls_token_head, std=0.02)
                trunc_normal_(self.cls_token_tail, std=0.02)
            else:
                trunc_normal_(self.cls_token, std=0.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

        # MLBN编码器特殊初始化
        if hasattr(self, "encoder") and hasattr(self.encoder, "apply_encoder_initialization"):
            print("应用MLBN编码器特殊初始化...")
            self.encoder.apply_encoder_initialization()

    def _get_abs_pos_embed(self, seq_len, device, dtype):
        if not self.if_abs_pos_embed:
            return None
        grid_h, grid_w = self.patch_embed.grid_size
        token_area = grid_h * grid_w
        if token_area != seq_len:
            # fallback: assume square if mismatch (shouldn't happen with fixed input)
            size = int(math.sqrt(seq_len))
            grid_h = grid_w = size
        extra_tokens = self.num_tokens
        if extra_tokens > 0:
            cls_pos = self.pos_embed[:, :extra_tokens, :]
            pos_tokens = self.pos_embed[:, extra_tokens:, :]
        else:
            cls_pos = self.pos_embed[:, :0, :]
            pos_tokens = self.pos_embed
        if grid_h == self.patch_embed.grid_size[0] and grid_w == self.patch_embed.grid_size[1]:
            return self.pos_embed
        pos_tokens = pos_tokens.reshape(1, self.patch_embed.grid_size[0], self.patch_embed.grid_size[1], -1)
        pos_tokens = pos_tokens.permute(0, 3, 1, 2)
        pos_tokens = F.interpolate(pos_tokens, size=(grid_h, grid_w), mode="bicubic", align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, -1)
        return torch.cat((cls_pos, pos_tokens), dim=1).to(device=device, dtype=dtype)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        # MLBN编码器不需要inference cache，但保持与原始接口一致
        # 原始实现返回每个layer的cache，这里返回空字典
        return {}

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "pos_embed",
            "cls_token",
            "dist_token",
            "cls_token_head",
            "cls_token_tail",
        }

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(
        self,
        x,
        inference_params=None,
        if_random_cls_token_position=False,
        if_random_token_rank=False,
    ):
        # PatchEmbed 输出: (2, B, M, D) - 原图和中心对称图的patch embedding
        x = self.patch_embed(x)  # (2, B, M, D)
        _, B, M, _ = x.shape
        token_position = None

        # 对两个分量分别添加 CLS token
        if self.if_cls_token:
            if self.use_double_cls_token:
                cls_token_head = self.cls_token_head.expand(B, -1, -1)
                cls_token_tail = self.cls_token_tail.expand(B, -1, -1)
                # 对两个分量分别添加 cls token
                x0 = torch.cat((cls_token_head, x[0], cls_token_tail), dim=1)
                x1 = torch.cat((cls_token_head, x[1], cls_token_tail), dim=1)
                x = torch.stack([x0, x1], dim=0)
                M = x.shape[2]
                token_position = [0, M - 1]
            else:
                cls_token = self.cls_token.expand(B, -1, -1)
                if if_random_cls_token_position:
                    token_position = random.randint(0, M)
                    x0 = torch.cat(
                        (
                            x[0, :, :token_position, :],
                            cls_token,
                            x[0, :, token_position:, :],
                        ),
                        dim=1,
                    )
                    x1 = torch.cat(
                        (
                            x[1, :, :token_position, :],
                            cls_token,
                            x[1, :, token_position:, :],
                        ),
                        dim=1,
                    )
                    x = torch.stack([x0, x1], dim=0)
                elif self.cls_strategy == "middle":
                    if M % 2 != 0:
                        raise ValueError(
                            "cls_strategy='middle' requires even sequence length before inserting CLS token"
                        )
                    token_position = M // 2
                    x0 = torch.cat(
                        (
                            x[0, :, :token_position, :],
                            cls_token,
                            x[0, :, token_position:, :],
                        ),
                        dim=1,
                    )
                    x1 = torch.cat(
                        (
                            x[1, :, :token_position, :],
                            cls_token,
                            x[1, :, token_position:, :],
                        ),
                        dim=1,
                    )
                    x = torch.stack([x0, x1], dim=0)
                elif self.cls_strategy == "last":
                    token_position = M
                    x0 = torch.cat((x[0], cls_token), dim=1)
                    x1 = torch.cat((x[1], cls_token), dim=1)
                    x = torch.stack([x0, x1], dim=0)
                else:  # default first
                    token_position = 0
                    x0 = torch.cat((cls_token, x[0]), dim=1)
                    x1 = torch.cat((cls_token, x[1]), dim=1)
                    x = torch.stack([x0, x1], dim=0)
                M = x.shape[2]

        # 对两个分量分别添加位置编码
        if self.if_abs_pos_embed:
            pos_embed = self._get_abs_pos_embed(
                seq_len=self.patch_embed.grid_size[0] * self.patch_embed.grid_size[1],
                device=x.device,
                dtype=x.dtype,
            )
            if pos_embed is not None:
                x = x + pos_embed  # 广播到两个分量
            x = torch.stack([self.pos_drop(x[0]), self.pos_drop(x[1])], dim=0)

        if if_random_token_rank:
            # 生成随机 shuffle 索引
            shuffle_indices = torch.randperm(M)

            # 对两个分量执行相同的 shuffle
            x = x[:, :, shuffle_indices, :]

            if self.if_cls_token and token_position is not None:
                if isinstance(token_position, list):
                    token_position = [torch.where(shuffle_indices == pos)[0].item() for pos in token_position]
                else:
                    token_position = torch.where(shuffle_indices == token_position)[0].item()

        if_flip_img_sequences = False
        if self.flip_img_sequences_ratio > 0 and (self.flip_img_sequences_ratio - random.random()) > 1e-5:
            x = x.flip([2])  # 在序列维度上翻转
            if_flip_img_sequences = True

        # MLBN编码器实现 - 输入已经是 (2, B, M, D) 格式
        residual = None
        hidden_states = x  # (2, B, M, D)

        # 保持与原始Vision Mamba完全一致的rope和bidirectional处理逻辑
        if not self.if_bidirectional:
            if if_flip_img_sequences and self.if_rope:
                hidden_states = hidden_states.flip([2])
                if residual is not None:
                    residual = residual.flip([2])

            # rope处理（对两个分量）
            if self.if_rope:
                hidden_states = torch.stack([self.rope(hidden_states[0]), self.rope(hidden_states[1])], dim=0)
                if residual is not None and self.if_rope_residual:
                    residual = torch.stack([self.rope(residual[0]), self.rope(residual[1])], dim=0)

            if if_flip_img_sequences and self.if_rope:
                hidden_states = hidden_states.flip([2])
                if residual is not None:
                    residual = residual.flip([2])

            # 使用MLBN编码器处理 - 输入 (2, B, M, D)
            hidden_states = self.encoder(hidden_states)
        else:
            # bidirectional模式
            if self.if_rope:
                hidden_states = torch.stack([self.rope(hidden_states[0]), self.rope(hidden_states[1])], dim=0)
                if residual is not None and self.if_rope_residual:
                    residual = torch.stack([self.rope(residual[0]), self.rope(residual[1])], dim=0)

            # MLBN编码器内部已经实现了双向处理
            hidden_states = self.encoder(hidden_states)

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        # return only cls token if it exists
        if self.if_cls_token:
            if self.use_double_cls_token:
                return (hidden_states[:, token_position[0], :] + hidden_states[:, token_position[1], :]) / 2
            else:
                return hidden_states[:, token_position, :]

        if self.final_pool_type == "none":
            return hidden_states[:, -1, :]
        elif self.final_pool_type == "mean":
            return hidden_states.mean(dim=1)
        elif self.final_pool_type == "max" or self.final_pool_type == "all":
            return hidden_states
        else:
            raise NotImplementedError

    def forward(
        self,
        x,
        return_features=False,
        inference_params=None,
        if_random_cls_token_position=False,
        if_random_token_rank=False,
    ):
        x = self.forward_features(
            x,
            inference_params,
            if_random_cls_token_position=if_random_cls_token_position,
            if_random_token_rank=if_random_token_rank,
        )
        if return_features:
            return x
        x = self.head(x)
        if self.final_pool_type == "max":
            x = x.max(dim=1)[0]
        return x


@register_model
def vim_tiny_patch16_224_mlbn(pretrained=False, drop_path_rate=0.18, **kwargs):
    model = VisionMambaMLBN(
        patch_size=16,
        embed_dim=192,  # ViT-Tiny标准配置
        depth=18,  # 介于ViT(12)和ViM(24)之间，适合MLBN架构
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        final_pool_type="mean",
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        bimamba_type="v2",
        cls_strategy="first",
        if_divide_out=True,
        drop_rate=0.1,  # 降低dropout（模型容量增加）
        drop_path_rate=drop_path_rate,
        # MLBN编码器参数（Vision Mamba标准配置）
        mlbn_kernel_size=4,
        mlbn_d_state=16,  # ViM标准
        mlbn_d_conv=4,  # ViM标准
        mlbn_expand=2,  # ViM标准
        mlbn_head_dim=16,  # ViM标准
        mlbn_dropout_rate1=0.08,
        mlbn_num_heads=3,  # 192/3=64，标准attention head_dim配置
        mlbn_dropout_rate2=0.08,
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(url="to.do", map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def vim_tiny_patch16_stride8_224_mlbn(pretrained=False, **kwargs):
    model = VisionMambaMLBN(
        patch_size=16,
        stride=8,
        embed_dim=192,  # ViT-Tiny标准配置
        depth=18,  # 与patch16版本保持一致
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        final_pool_type="mean",
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        bimamba_type="v2",
        cls_strategy="middle",
        if_divide_out=True,
        mlbn_num_heads=3,  # 192/3=64，标准head_dim配置
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(url="to.do", map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def vim_small_patch16_224_mlbn(pretrained=False, drop_path_rate=0.2, **kwargs):
    model = VisionMambaMLBN(
        patch_size=16,
        embed_dim=384,  # ViT-Small标准配置
        depth=18,  # 介于ViT(12)和ViM(24)之间
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        final_pool_type="mean",
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        bimamba_type="v2",
        cls_strategy="middle",
        if_divide_out=True,
        drop_path_rate=drop_path_rate,
        mlbn_num_heads=6,  # 384/6=64，标准head_dim配置
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(url="to.do", map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def vim_small_patch16_stride8_224_mlbn(pretrained=False, **kwargs):
    model = VisionMambaMLBN(
        patch_size=16,
        stride=8,
        embed_dim=384,  # ViT-Small标准配置
        depth=18,  # 与patch16版本保持一致
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        final_pool_type="mean",
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        bimamba_type="v2",
        cls_strategy="middle",
        if_divide_out=True,
        mlbn_num_heads=6,  # 384/6=64，标准head_dim配置
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(url="to.do", map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def vim_base_patch16_224_mlbn(pretrained=False, drop_path_rate=0.25, **kwargs):
    model = VisionMambaMLBN(
        patch_size=16,
        embed_dim=768,  # ViT-Base标准配置
        d_state=16,
        depth=18,  # 介于ViT(12)和ViM(24)之间
        drop_path_rate=drop_path_rate,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        final_pool_type="mean",
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        bimamba_type="v2",
        cls_strategy="middle",
        if_divide_out=True,
        mlbn_num_heads=12,  # 768/12=64，标准head_dim配置
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(url="to.do", map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model
