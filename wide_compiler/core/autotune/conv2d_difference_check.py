"""
Deep investigation: Why does grouped conv have worse numerical accuracy?

Hypothesis testing:
1. Is it the packing/unpacking of tensors?
2. Is it the grouped conv kernel itself?
3. Does it scale with tensor magnitude?
4. Is it consistent or random?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
device = 'cuda'

print("=" * 70)
print("GROUPED CONV NUMERICAL ACCURACY INVESTIGATION")
print("=" * 70)

N, B = 10, 8
C_in, C_out = 64, 64
H, W = 56, 56
kernel = 3
padding = 1

# =============================================================================
# TEST 1: Is it the grouped conv kernel or our packing?
# =============================================================================
print("\n" + "=" * 70)
print("TEST 1: Grouped kernel vs packing")
print("=" * 70)

# Create identical weights
weights = [torch.randn(C_out, C_in, kernel, kernel, device=device) for _ in range(N)]
biases = [torch.randn(C_out, device=device) for _ in range(N)]
x_list = [torch.randn(B, C_in, H, W, device=device) for _ in range(N)]

# Method A: Sequential F.conv2d (reference)
ref_outputs = []
for i in range(N):
    out = F.conv2d(x_list[i], weights[i], biases[i], padding=padding)
    ref_outputs.append(out)

# Method B: Grouped conv with our packing
grouped_w = torch.cat(weights, dim=0)  # [N*C_out, C_in, K, K]
grouped_b = torch.cat(biases, dim=0)  # [N*C_out]

# Pack inputs: [B, N*C_in, H, W]
x_packed = torch.stack(x_list, dim=1).view(B, N * C_in, H, W)

grouped_out = F.conv2d(x_packed, grouped_w, grouped_b, padding=padding, groups=N)
# Unpack: [B, N*C_out, H, W] -> list of [B, C_out, H, W]
grouped_outputs = grouped_out.view(B, N, C_out, H, W).unbind(dim=1)

# Compare
print("\nPer-model max absolute difference:")
for i in range(N):
    diff = (ref_outputs[i] - grouped_outputs[i]).abs().max().item()
    print(f"  Model {i}: {diff:.6e}")

overall_max = max((ref_outputs[i] - grouped_outputs[i]).abs().max().item() for i in range(N))
print(f"\nOverall max diff: {overall_max:.6e}")

# =============================================================================
# TEST 2: Is grouped conv inherently less accurate?
# =============================================================================
print("\n" + "=" * 70)
print("TEST 2: Single grouped conv vs N=1 standard conv")
print("=" * 70)

# Same conv, but one uses groups=1, other uses standard
w = torch.randn(C_out, C_in, kernel, kernel, device=device)
b = torch.randn(C_out, device=device)
x = torch.randn(B, C_in, H, W, device=device)

out_standard = F.conv2d(x, w, b, padding=padding)
out_grouped = F.conv2d(x, w, b, padding=padding, groups=1)  # groups=1 is same as standard

diff = (out_standard - out_grouped).abs().max().item()
print(f"Standard vs groups=1: {diff:.6e}")
print("(Should be 0.0 - same kernel)")

# =============================================================================
# TEST 3: Does the difference come from weight layout?
# =============================================================================
print("\n" + "=" * 70)
print("TEST 3: Weight layout investigation")
print("=" * 70)

# For grouped conv, cuDNN may use different memory access patterns
# Let's check if contiguous vs non-contiguous matters

# Force weights to be contiguous
grouped_w_contig = grouped_w.contiguous()
grouped_b_contig = grouped_b.contiguous()
x_packed_contig = x_packed.contiguous()

out_contig = F.conv2d(x_packed_contig, grouped_w_contig, grouped_b_contig,
                      padding=padding, groups=N)

diff_contig = (grouped_out - out_contig).abs().max().item()
print(f"Contiguous vs original: {diff_contig}")

# =============================================================================
# TEST 4: Does magnitude affect the error?
# =============================================================================
print("\n" + "=" * 70)
print("TEST 4: Error scaling with tensor magnitude")
print("=" * 70)

for scale in [0.01, 1.0, 100.0]:
    weights_scaled = [w * scale for w in weights]
    x_scaled = [xi * scale for xi in x_list]

    # Reference
    ref_scaled = []
    for i in range(N):
        out = F.conv2d(x_scaled[i], weights_scaled[i], biases[i] * scale, padding=padding)
        ref_scaled.append(out)

    # Grouped
    gw = torch.cat(weights_scaled, dim=0)
    gb = torch.cat([bi * scale for bi in biases], dim=0)
    xp = torch.stack(x_scaled, dim=1).view(B, N * C_in, H, W)

    gout = F.conv2d(xp, gw, gb, padding=padding, groups=N)
    gout_list = gout.view(B, N, C_out, H, W).unbind(dim=1)

    max_diff = max((ref_scaled[i] - gout_list[i]).abs().max().item() for i in range(N))
    max_val = max(ref_scaled[i].abs().max().item() for i in range(N))
    rel_diff = max_diff / max_val if max_val > 0 else 0

    print(f"  Scale {scale:6.2f}: abs_diff={max_diff:.6e}, rel_diff={rel_diff:.6e}")

# =============================================================================
# TEST 5: Is it deterministic?
# =============================================================================
print("\n" + "=" * 70)
print("TEST 5: Determinism check")
print("=" * 70)

results = []
for trial in range(5):
    out = F.conv2d(x_packed, grouped_w, grouped_b, padding=padding, groups=N)
    results.append(out.clone())

all_same = all(torch.equal(results[0], results[i]) for i in range(1, 5))
print(f"All 5 runs identical: {all_same}")

if not all_same:
    for i in range(1, 5):
        diff = (results[0] - results[i]).abs().max().item()
        print(f"  Run 0 vs {i}: {diff:.6e}")

# =============================================================================
# TEST 6: cuDNN algorithm selection
# =============================================================================
print("\n" + "=" * 70)
print("TEST 6: cuDNN benchmark mode effect")
print("=" * 70)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

out_deterministic = F.conv2d(x_packed, grouped_w, grouped_b, padding=padding, groups=N)
det_outputs = out_deterministic.view(B, N, C_out, H, W).unbind(dim=1)

max_diff_det = max((ref_outputs[i] - det_outputs[i]).abs().max().item() for i in range(N))
print(f"With deterministic=True: {max_diff_det:.6e}")

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

out_benchmark = F.conv2d(x_packed, grouped_w, grouped_b, padding=padding, groups=N)
bench_outputs = out_benchmark.view(B, N, C_out, H, W).unbind(dim=1)

max_diff_bench = max((ref_outputs[i] - bench_outputs[i]).abs().max().item() for i in range(N))
print(f"With benchmark=True: {max_diff_bench:.6e}")

# =============================================================================
# TEST 7: CPU vs CUDA
# =============================================================================
print("\n" + "=" * 70)
print("TEST 7: CPU computation (no cuDNN)")
print("=" * 70)

# Move to CPU
weights_cpu = [w.cpu() for w in weights]
biases_cpu = [b.cpu() for b in biases]
x_list_cpu = [x.cpu() for x in x_list]

# Reference on CPU
ref_cpu = []
for i in range(N):
    out = F.conv2d(x_list_cpu[i], weights_cpu[i], biases_cpu[i], padding=padding)
    ref_cpu.append(out)

# Grouped on CPU
gw_cpu = torch.cat(weights_cpu, dim=0)
gb_cpu = torch.cat(biases_cpu, dim=0)
xp_cpu = torch.stack(x_list_cpu, dim=1).view(B, N * C_in, H, W)

gout_cpu = F.conv2d(xp_cpu, gw_cpu, gb_cpu, padding=padding, groups=N)
gout_cpu_list = gout_cpu.view(B, N, C_out, H, W).unbind(dim=1)

max_diff_cpu = max((ref_cpu[i] - gout_cpu_list[i]).abs().max().item() for i in range(N))
print(f"CPU grouped vs sequential: {max_diff_cpu:.6e}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)