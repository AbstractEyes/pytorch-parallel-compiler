"""
Compare Conv1d grouped vs Conv2d grouped accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
device = 'cuda'

print("=" * 60)
print("Conv1d vs Conv2d GROUPED ACCURACY")
print("=" * 60)

N = 10
B = 32
C_in, C_out = 512, 256

# =============================================================================
# Conv2d grouped (spatial 1x1 to simulate linear)
# =============================================================================
print("\n--- Conv2d grouped (1x1 spatial) ---")

weights_2d = [torch.randn(C_out, C_in, 1, 1, device=device) for _ in range(N)]
biases = [torch.randn(C_out, device=device) for _ in range(N)]
x_list = [torch.randn(B, C_in, 1, 1, device=device) for _ in range(N)]

# Reference
ref_2d = [F.conv2d(x_list[i], weights_2d[i], biases[i]) for i in range(N)]

# Grouped
gw_2d = torch.cat(weights_2d, dim=0)
gb = torch.cat(biases, dim=0)
xp_2d = torch.stack(x_list, dim=1).view(B, N * C_in, 1, 1)

gout_2d = F.conv2d(xp_2d, gw_2d, gb, groups=N)
gout_2d_list = gout_2d.view(B, N, C_out, 1, 1).unbind(dim=1)

max_diff_2d = max((ref_2d[i] - gout_2d_list[i]).abs().max().item() for i in range(N))
print(f"Max diff: {max_diff_2d:.6e}")

# =============================================================================
# Conv1d grouped (L=1 to simulate linear)
# =============================================================================
print("\n--- Conv1d grouped (L=1) ---")

weights_1d = [torch.randn(C_out, C_in, 1, device=device) for _ in range(N)]
x_list_1d = [torch.randn(B, C_in, 1, device=device) for _ in range(N)]

# Reference
ref_1d = [F.conv1d(x_list_1d[i], weights_1d[i], biases[i]) for i in range(N)]

# Grouped
gw_1d = torch.cat(weights_1d, dim=0)
xp_1d = torch.stack(x_list_1d, dim=1).view(B, N * C_in, 1)

gout_1d = F.conv1d(xp_1d, gw_1d, gb, groups=N)
gout_1d_list = gout_1d.view(B, N, C_out, 1).unbind(dim=1)

max_diff_1d = max((ref_1d[i] - gout_1d_list[i]).abs().max().item() for i in range(N))
print(f"Max diff: {max_diff_1d:.6e}")

# =============================================================================
# F.linear vs Conv1d(k=1) for same weights
# =============================================================================
print("\n--- F.linear vs Conv1d(k=1) same weights ---")

weights_lin = [torch.randn(C_out, C_in, device=device) for _ in range(N)]
x_flat = [torch.randn(B, C_in, device=device) for _ in range(N)]

# F.linear reference
ref_lin = [F.linear(x_flat[i], weights_lin[i], biases[i]) for i in range(N)]

# Conv1d with same weights
weights_as_conv = [w.unsqueeze(-1) for w in weights_lin]  # [out, in, 1]
x_as_conv = [x.unsqueeze(-1) for x in x_flat]  # [B, in, 1]

conv1d_out = [F.conv1d(x_as_conv[i], weights_as_conv[i], biases[i]).squeeze(-1) for i in range(N)]

max_diff_lin_conv = max((ref_lin[i] - conv1d_out[i]).abs().max().item() for i in range(N))
print(f"F.linear vs Conv1d (sequential): {max_diff_lin_conv:.6e}")

# Now grouped
gw_lin = torch.cat(weights_as_conv, dim=0)
xp_lin = torch.cat(x_as_conv, dim=1)

gout_lin = F.conv1d(xp_lin, gw_lin, gb, groups=N).squeeze(-1)
gout_lin_list = gout_lin.view(B, N, C_out).unbind(dim=1)

max_diff_lin_grouped = max((ref_lin[i] - gout_lin_list[i]).abs().max().item() for i in range(N))
print(f"F.linear vs Conv1d (grouped):    {max_diff_lin_grouped:.6e}")

# =============================================================================
# Direct einsum comparison
# =============================================================================
print("\n--- Einsum vs F.linear ---")

stacked_w = torch.stack(weights_lin)  # [N, out, in]
stacked_b = torch.stack(biases)       # [N, out]
x_stacked = torch.stack(x_flat)       # [N, B, in]

ein_out = torch.einsum('noi,nbi->nbo', stacked_w, x_stacked) + stacked_b.unsqueeze(1)
ein_list = ein_out.unbind(dim=0)

max_diff_ein = max((ref_lin[i] - ein_list[i]).abs().max().item() for i in range(N))
print(f"F.linear vs einsum: {max_diff_ein:.6e}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"""
Conv2d grouped:     {max_diff_2d:.2e}
Conv1d grouped:     {max_diff_1d:.2e}  
Conv1d for Linear:  {max_diff_lin_grouped:.2e}
Einsum for Linear:  {max_diff_ein:.2e}
""")