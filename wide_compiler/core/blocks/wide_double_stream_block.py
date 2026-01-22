"""
WideDoubleStreamBlock - N parallel Flux-style double-stream transformer blocks.

Complete MMDiT block with joint attention over text and image streams.
This is the core building block of Flux and similar multi-modal diffusion models.

Architecture (per stream):
    txt, img -> adaptive norm (modulated by vec)
             -> joint attention (both streams attend to txt+img)
             -> MLP
             -> residual connection

Expected speedup: 8-12x (dominated by attention and MLP)

Strategies:
- 'fused': Fused operations throughout (FASTEST)
- 'sequential': N separate blocks (baseline)

Input/Output Format (v0.7.0):
- txt:  [N, B, L, hidden_size]  (N-first, text sequence)
- img:  [N, B, S, hidden_size]  (N-first, image sequence)
- vec:  [N, B, emb_size]        (N-first, conditioning vector)
- RoPE: Optional [B, S, head_dim] (applied to image only)
- Output: (txt_out, img_out) both [N, B, *, hidden_size]

Copyright 2025 AbstractPhil
Apache 2.0 License
"""

from typing import List, Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .wide_joint_attention import WideJointAttention
from .wide_mlp import WideMLP


class WideDoubleStreamBlock(nn.Module):
    """
    N parallel double-stream transformer blocks (Flux-style MMDiT).

    Two streams (text and image) with:
    - Adaptive normalization conditioned on vec
    - Joint attention (both attend to concatenated sequence)
    - Feed-forward MLPs
    - Residual connections
    """

    def __init__(
        self,
        n: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: float = 4.0,
        bias: bool = False,
        strategy: str = 'fused',
    ):
        super().__init__()
        self.n = n
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.mlp_ratio = mlp_ratio
        self._strategy = strategy

        intermediate_size = int(hidden_size * mlp_ratio)

        # Text stream adaptive norm (simple scale + shift from vec)
        self.txt_norm_weight = nn.Parameter(torch.ones(n, hidden_size))
        self.txt_norm_bias = nn.Parameter(torch.zeros(n, hidden_size))
        self.txt_mod_weight = nn.Parameter(torch.empty(n, 2 * hidden_size, hidden_size))
        self.txt_mod_bias = nn.Parameter(torch.zeros(n, 2 * hidden_size))

        # Image stream adaptive norm
        self.img_norm_weight = nn.Parameter(torch.ones(n, hidden_size))
        self.img_norm_bias = nn.Parameter(torch.zeros(n, hidden_size))
        self.img_mod_weight = nn.Parameter(torch.empty(n, 2 * hidden_size, hidden_size))
        self.img_mod_bias = nn.Parameter(torch.zeros(n, 2 * hidden_size))

        # Joint attention
        self.attn = WideJointAttention(
            n=n,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            bias=bias,
            strategy=strategy,
        )

        # Text MLP
        self.txt_mlp = WideMLP(
            n=n,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation='gelu',
            bias=bias,
            strategy=strategy,
        )

        # Image MLP
        self.img_mlp = WideMLP(
            n=n,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation='gelu',
            bias=bias,
            strategy=strategy,
        )

        self.eps = 1e-6
        self._reset_parameters()

    @property
    def strategy(self) -> str:
        return self._strategy

    def _reset_parameters(self):
        """Initialize weights."""
        for i in range(self.n):
            nn.init.xavier_uniform_(self.txt_mod_weight[i])
            nn.init.xavier_uniform_(self.img_mod_weight[i])

    def forward(
        self,
        txt: Tensor,
        img: Tensor,
        vec: Tensor,
        rope: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass with N-first format.

        Args:
            txt: [N, B, L, hidden_size] - text tokens
            img: [N, B, S, hidden_size] - image tokens
            vec: [N, B, emb_size] - conditioning vector (timestep/text embedding)
            rope: Optional [B, S, head_dim] - RoPE for image only

        Returns:
            (txt_out, img_out):
                txt_out: [N, B, L, hidden_size]
                img_out: [N, B, S, hidden_size]
        """
        if self._strategy == 'fused':
            return self._forward_fused(txt, img, vec, rope)
        else:
            return self._forward_sequential(txt, img, vec, rope)

    def _forward_fused(
        self,
        txt: Tensor,
        img: Tensor,
        vec: Tensor,
        rope: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Fused double-stream block."""
        N, B, L, H = txt.shape
        _, _, S, _ = img.shape

        # === Text stream adaptive norm ===
        # Compute modulation parameters from vec
        txt_mod = torch.einsum('nbe,nme->nbm', vec, self.txt_mod_weight)
        txt_mod = txt_mod + self.txt_mod_bias.unsqueeze(1)
        txt_scale, txt_shift = txt_mod.chunk(2, dim=-1)  # Each [N, B, H]

        # LayerNorm
        txt_mean = txt.mean(dim=-1, keepdim=True)
        txt_var = txt.var(dim=-1, keepdim=True, unbiased=False)
        txt_norm = (txt - txt_mean) / torch.sqrt(txt_var + self.eps)

        # Apply learned norm params
        weight_txt = self.txt_norm_weight.view(N, 1, 1, H)
        bias_txt = self.txt_norm_bias.view(N, 1, 1, H)
        txt_norm = txt_norm * weight_txt + bias_txt

        # Apply adaptive modulation
        txt_scale = txt_scale.unsqueeze(2)  # [N, B, 1, H]
        txt_shift = txt_shift.unsqueeze(2)
        txt_norm = txt_norm * (1 + txt_scale) + txt_shift

        # === Image stream adaptive norm ===
        img_mod = torch.einsum('nbe,nme->nbm', vec, self.img_mod_weight)
        img_mod = img_mod + self.img_mod_bias.unsqueeze(1)
        img_scale, img_shift = img_mod.chunk(2, dim=-1)

        img_mean = img.mean(dim=-1, keepdim=True)
        img_var = img.var(dim=-1, keepdim=True, unbiased=False)
        img_norm = (img - img_mean) / torch.sqrt(img_var + self.eps)

        weight_img = self.img_norm_weight.view(N, 1, 1, H)
        bias_img = self.img_norm_bias.view(N, 1, 1, H)
        img_norm = img_norm * weight_img + bias_img

        img_scale = img_scale.unsqueeze(2)
        img_shift = img_shift.unsqueeze(2)
        img_norm = img_norm * (1 + img_scale) + img_shift

        # === Joint attention ===
        txt_attn, img_attn = self.attn(txt_norm, img_norm, rope)

        # === MLPs with residual ===
        txt_out = txt + txt_attn + self.txt_mlp(txt_norm)
        img_out = img + img_attn + self.img_mlp(img_norm)

        return txt_out, img_out

    def _forward_sequential(
        self,
        txt: Tensor,
        img: Tensor,
        vec: Tensor,
        rope: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Sequential double-stream block (baseline)."""
        N, B, L, H = txt.shape
        _, _, S, _ = img.shape

        txt_outputs = []
        img_outputs = []

        for i in range(N):
            txt_i = txt[i]  # [B, L, H]
            img_i = img[i]  # [B, S, H]
            vec_i = vec[i]  # [B, E]

            # Text adaptive norm (weights already in [out, in] format)
            txt_mod_i = F.linear(vec_i, self.txt_mod_weight[i], self.txt_mod_bias[i])
            txt_scale_i, txt_shift_i = txt_mod_i.chunk(2, dim=-1)

            txt_mean_i = txt_i.mean(dim=-1, keepdim=True)
            txt_var_i = txt_i.var(dim=-1, keepdim=True, unbiased=False)
            txt_norm_i = (txt_i - txt_mean_i) / torch.sqrt(txt_var_i + self.eps)
            txt_norm_i = txt_norm_i * self.txt_norm_weight[i] + self.txt_norm_bias[i]
            txt_norm_i = txt_norm_i * (1 + txt_scale_i.unsqueeze(1)) + txt_shift_i.unsqueeze(1)

            # Image adaptive norm (weights already in [out, in] format)
            img_mod_i = F.linear(vec_i, self.img_mod_weight[i], self.img_mod_bias[i])
            img_scale_i, img_shift_i = img_mod_i.chunk(2, dim=-1)

            img_mean_i = img_i.mean(dim=-1, keepdim=True)
            img_var_i = img_i.var(dim=-1, keepdim=True, unbiased=False)
            img_norm_i = (img_i - img_mean_i) / torch.sqrt(img_var_i + self.eps)
            img_norm_i = img_norm_i * self.img_norm_weight[i] + self.img_norm_bias[i]
            img_norm_i = img_norm_i * (1 + img_scale_i.unsqueeze(1)) + img_shift_i.unsqueeze(1)

            # Joint attention - manually compute using weights at index i
            # (cannot call _forward_sequential as it would use wrong index)
            txt_attn_i, img_attn_i = self._compute_joint_attn_i(i, txt_norm_i, img_norm_i, rope)

            # MLPs - manually compute using weights at index i
            txt_mlp_i = self._compute_mlp_i(self.txt_mlp, i, txt_norm_i)
            img_mlp_i = self._compute_mlp_i(self.img_mlp, i, img_norm_i)

            txt_out_i = txt_i + txt_attn_i + txt_mlp_i
            img_out_i = img_i + img_attn_i + img_mlp_i

            txt_outputs.append(txt_out_i)
            img_outputs.append(img_out_i)

        return torch.stack(txt_outputs, dim=0), torch.stack(img_outputs, dim=0)

    def _compute_joint_attn_i(
        self,
        i: int,
        txt_norm: Tensor,
        img_norm: Tensor,
        rope: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Compute joint attention for index i using manual weight access."""
        B, L, H = txt_norm.shape
        _, S, _ = img_norm.shape

        # QKV projections using weights at index i
        txt_qkv = F.linear(txt_norm, self.attn.txt_qkv_weight[i], self.attn.txt_qkv_bias[i] if self.attn.txt_qkv_bias is not None else None)
        img_qkv = F.linear(img_norm, self.attn.img_qkv_weight[i], self.attn.img_qkv_bias[i] if self.attn.img_qkv_bias is not None else None)

        # Reshape for multi-head
        txt_qkv = txt_qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        img_qkv = img_qkv.reshape(B, S, 3, self.num_heads, self.head_dim)

        txt_q, txt_k, txt_v = txt_qkv.permute(2, 0, 3, 1, 4)
        img_q, img_k, img_v = img_qkv.permute(2, 0, 3, 1, 4)

        # Concatenate K, V
        k = torch.cat([txt_k, img_k], dim=2)
        v = torch.cat([txt_v, img_v], dim=2)

        # Attention
        scale = self.head_dim ** -0.5
        txt_attn = F.scaled_dot_product_attention(txt_q, k, v, scale=scale)
        img_attn = F.scaled_dot_product_attention(img_q, k, v, scale=scale)

        # Reshape back
        txt_attn = txt_attn.transpose(1, 2).reshape(B, L, -1)
        img_attn = img_attn.transpose(1, 2).reshape(B, S, -1)

        # Output projections using weights at index i
        txt_out = F.linear(txt_attn, self.attn.txt_out_weight[i], self.attn.txt_out_bias[i] if self.attn.txt_out_bias is not None else None)
        img_out = F.linear(img_attn, self.attn.img_out_weight[i], self.attn.img_out_bias[i] if self.attn.img_out_bias is not None else None)

        return txt_out, img_out

    def _compute_mlp_i(self, mlp: 'WideMLP', i: int, x: Tensor) -> Tensor:
        """Compute MLP forward for index i using manual weight access."""
        # FC1
        h = F.linear(x, mlp.fc1_weight[i], mlp.fc1_bias[i] if mlp.fc1_bias is not None else None)
        # Activation
        h = mlp._activation(h)
        # FC2
        out = F.linear(h, mlp.fc2_weight[i], mlp.fc2_bias[i] if mlp.fc2_bias is not None else None)
        return out

    @classmethod
    def from_modules(cls, modules: List[nn.Module], strategy: str = 'fused') -> 'WideDoubleStreamBlock':
        """
        Create from N existing DoubleStreamBlock modules.

        Expects modules with:
        - .txt_norm, .img_norm: normalization layers
        - .txt_mod, .img_mod: modulation linear layers
        - .attn: joint attention module
        - .txt_mlp, .img_mlp: MLP modules
        - .num_heads, .head_dim: attention config
        """
        n = len(modules)
        t = modules[0]

        hidden_size = t.txt_norm.normalized_shape[0]
        num_heads = t.attn.num_heads
        head_dim = t.attn.head_dim

        # Detect MLP ratio
        mlp_ratio = 4.0  # Default
        if hasattr(t, 'txt_mlp'):
            mlp = t.txt_mlp
            if isinstance(mlp, nn.Sequential):
                fc1 = mlp[0]
            else:
                fc1 = getattr(mlp, 'fc1', getattr(mlp, 'linear1', None))
            if fc1 is not None:
                mlp_ratio = fc1.out_features / fc1.in_features

        wide = cls(
            n=n,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            mlp_ratio=mlp_ratio,
            bias=(t.txt_mod.bias is not None),
            strategy=strategy,
        )

        # Copy weights
        with torch.no_grad():
            for i, m in enumerate(modules):
                # Text norm
                wide.txt_norm_weight[i] = m.txt_norm.weight
                wide.txt_norm_bias[i] = m.txt_norm.bias

                # Image norm
                wide.img_norm_weight[i] = m.img_norm.weight
                wide.img_norm_bias[i] = m.img_norm.bias

                # Text modulation
                wide.txt_mod_weight[i] = m.txt_mod.weight
                if m.txt_mod.bias is not None:
                    wide.txt_mod_bias[i] = m.txt_mod.bias

                # Image modulation
                wide.img_mod_weight[i] = m.img_mod.weight
                if m.img_mod.bias is not None:
                    wide.img_mod_bias[i] = m.img_mod.bias

            # Copy attention and MLP modules
            attn_modules = [m.attn for m in modules]
            txt_mlp_modules = [m.txt_mlp for m in modules]
            img_mlp_modules = [m.img_mlp for m in modules]

            wide.attn = WideJointAttention.from_modules(attn_modules, strategy=strategy)
            wide.txt_mlp = WideMLP.from_modules(txt_mlp_modules, strategy=strategy)
            wide.img_mlp = WideMLP.from_modules(img_mlp_modules, strategy=strategy)

        return wide

    def __repr__(self):
        return (f"WideDoubleStreamBlock({self.n}x[hidden={self.hidden_size}, "
                f"heads={self.num_heads}, head_dim={self.head_dim}, "
                f"mlp_ratio={self.mlp_ratio}], strategy={self._strategy})")

    # =========================================================================
    # Benchmark interface
    # =========================================================================

    BENCHMARK_STRATEGIES = ['baseline', 'fused', 'sequential']
    BENCHMARK_SWEEPS = {}
    _SWEEPS_INITIALIZED = False

    @classmethod
    def _get_sweep_params_class(cls):
        """Get SweepParams class."""
        try:
            from ..benchmark.benchmark_schema import SweepParams
            return SweepParams
        except ImportError:
            try:
                from wide_compiler.core.benchmark.benchmark_schema import SweepParams
                return SweepParams
            except ImportError:
                return None

    @classmethod
    def _init_benchmark_sweeps(cls):
        """Initialize sweep configs."""
        if cls._SWEEPS_INITIALIZED:
            return
        cls._SWEEPS_INITIALIZED = True

        SweepParams = cls._get_sweep_params_class()
        if SweepParams is None:
            return

        cls.BENCHMARK_SWEEPS = {
            'quick': SweepParams(
                n_values=[4, 8],
                batch_sizes=[4],
                seq_lengths=[128],
                d_model=[256],
                n_heads=[8],
            ),
            'ci': SweepParams(
                n_values=[4],
                batch_sizes=[4],
                seq_lengths=[64],
                d_model=[256],
                n_heads=[8],
            ),
        }

    @classmethod
    def benchmark_job(cls, preset='quick'):
        """Create benchmark job for WideDoubleStreamBlock."""
        cls._init_benchmark_sweeps()

        try:
            from ..benchmark.benchmark_schema import BenchmarkJob
        except ImportError:
            from wide_compiler.core.benchmark.benchmark_schema import BenchmarkJob

        if preset not in cls.BENCHMARK_SWEEPS:
            raise ValueError(f"Unknown preset '{preset}'. Available: {list(cls.BENCHMARK_SWEEPS.keys())}")

        sweep = cls.BENCHMARK_SWEEPS[preset]

        return BenchmarkJob(
            name=f'double_stream_block_{preset}',
            primitive='double_stream_block',
            strategies=cls.BENCHMARK_STRATEGIES,
            sweep=sweep,
            model_factory=cls._bench_model,
            input_factory=cls._bench_input,
            wide_factory=cls._bench_wide,
            validate_fn=cls._bench_validate,
        )

    @staticmethod
    def _bench_model(d_model, n_heads, **kwargs):
        """Create a single DoubleStreamBlock module."""
        hidden_size = d_model if isinstance(d_model, int) else d_model
        num_heads = n_heads if isinstance(n_heads, int) else n_heads
        head_dim = hidden_size // num_heads

        class SimpleJointAttn(nn.Module):
            """Simple joint attention matching expected structure."""
            def __init__(self):
                super().__init__()
                qkv_dim = 3 * num_heads * head_dim
                out_dim = num_heads * head_dim
                self.txt_qkv = nn.Linear(hidden_size, qkv_dim, bias=False)
                self.img_qkv = nn.Linear(hidden_size, qkv_dim, bias=False)
                self.txt_out = nn.Linear(out_dim, hidden_size, bias=False)
                self.img_out = nn.Linear(out_dim, hidden_size, bias=False)
                self.num_heads = num_heads
                self.head_dim = head_dim
                self.scale = head_dim ** -0.5

        class SimpleDoubleStreamBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_heads = num_heads
                self.head_dim = head_dim
                self.eps = 1e-6

                # Text stream norm + modulation
                self.txt_norm = nn.LayerNorm(hidden_size)
                self.txt_mod = nn.Linear(hidden_size, 2 * hidden_size, bias=True)

                # Image stream norm + modulation
                self.img_norm = nn.LayerNorm(hidden_size)
                self.img_mod = nn.Linear(hidden_size, 2 * hidden_size, bias=True)

                # Joint attention as sub-module
                self.attn = SimpleJointAttn()

                # MLPs
                mlp_hidden = hidden_size * 4
                self.txt_mlp = nn.Sequential(
                    nn.Linear(hidden_size, mlp_hidden, bias=False),
                    nn.GELU(),
                    nn.Linear(mlp_hidden, hidden_size, bias=False)
                )
                self.img_mlp = nn.Sequential(
                    nn.Linear(hidden_size, mlp_hidden, bias=False),
                    nn.GELU(),
                    nn.Linear(mlp_hidden, hidden_size, bias=False)
                )

            def forward(self, txt, img, vec, rope=None):
                B, L, H = txt.shape
                S = img.shape[1]

                # Adaptive norm
                txt_mod = self.txt_mod(vec)
                txt_scale, txt_shift = txt_mod.chunk(2, dim=-1)
                txt_norm = self.txt_norm(txt)
                txt_norm = txt_norm * (1 + txt_scale.unsqueeze(1)) + txt_shift.unsqueeze(1)

                img_mod = self.img_mod(vec)
                img_scale, img_shift = img_mod.chunk(2, dim=-1)
                img_norm = self.img_norm(img)
                img_norm = img_norm * (1 + img_scale.unsqueeze(1)) + img_shift.unsqueeze(1)

                # Joint attention (simplified, no RoPE)
                txt_qkv = self.attn.txt_qkv(txt_norm).reshape(B, L, 3, self.num_heads, self.head_dim)
                img_qkv = self.attn.img_qkv(img_norm).reshape(B, S, 3, self.num_heads, self.head_dim)

                txt_q, txt_k, txt_v = txt_qkv.permute(2, 0, 3, 1, 4)
                img_q, img_k, img_v = img_qkv.permute(2, 0, 3, 1, 4)

                k = torch.cat([txt_k, img_k], dim=2)
                v = torch.cat([txt_v, img_v], dim=2)

                txt_attn = F.scaled_dot_product_attention(txt_q, k, v, scale=self.attn.scale)
                img_attn = F.scaled_dot_product_attention(img_q, k, v, scale=self.attn.scale)

                txt_attn = txt_attn.transpose(1, 2).reshape(B, L, -1)
                img_attn = img_attn.transpose(1, 2).reshape(B, S, -1)

                txt_attn = self.attn.txt_out(txt_attn)
                img_attn = self.attn.img_out(img_attn)

                # MLP + residual
                txt_out = txt + txt_attn + self.txt_mlp(txt_norm)
                img_out = img + img_attn + self.img_mlp(img_norm)

                return txt_out, img_out

        return SimpleDoubleStreamBlock()

    @staticmethod
    def _bench_input(n, d_model, n_heads, batch_sizes, seq_lengths, device='cpu', **kwargs):
        """Create input tensors."""
        batch_size = batch_sizes if isinstance(batch_sizes, int) else batch_sizes
        hidden_size = d_model if isinstance(d_model, int) else d_model
        seq_len = seq_lengths if isinstance(seq_lengths, int) else seq_lengths

        # Split sequence: txt is 1/4, img is 3/4
        txt_seq_len = max(seq_len // 4, 16)
        img_seq_len = seq_len - txt_seq_len

        txt = torch.randn(batch_size, txt_seq_len, hidden_size, device=device)
        img = torch.randn(batch_size, img_seq_len, hidden_size, device=device)
        vec = torch.randn(batch_size, hidden_size, device=device)
        return (txt, img, vec, None)

    @classmethod
    def _bench_wide(cls, modules, strategy, **kwargs):
        """Create wide version."""
        if strategy == 'baseline':
            return None
        return cls.from_modules(modules, strategy=strategy)

    @staticmethod
    def _bench_validate(wide_output, baseline_outputs, rtol=1e-4, atol=1e-4):
        """Validate tuple outputs (txt, img)."""
        wide_txt, wide_img = wide_output
        baseline_txts = [out[0] for out in baseline_outputs]
        baseline_imgs = [out[1] for out in baseline_outputs]

        stacked_txt = torch.stack(baseline_txts, dim=0)
        stacked_img = torch.stack(baseline_imgs, dim=0)

        txt_match = torch.allclose(wide_txt, stacked_txt, rtol=rtol, atol=atol)
        if not txt_match:
            txt_diff = (wide_txt - stacked_txt).abs()
            return False, f"Text output mismatch: max_diff={txt_diff.max():.2e}, mean_diff={txt_diff.mean():.2e}"

        img_match = torch.allclose(wide_img, stacked_img, rtol=rtol, atol=atol)
        if not img_match:
            img_diff = (wide_img - stacked_img).abs()
            return False, f"Image output mismatch: max_diff={img_diff.max():.2e}, mean_diff={img_diff.mean():.2e}"

        return True, "OK"


__all__ = ['WideDoubleStreamBlock']
