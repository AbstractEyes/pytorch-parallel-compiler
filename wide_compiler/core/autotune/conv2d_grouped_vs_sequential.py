"""
Isolate where the 1e-3 error comes from.
"""

import torch
import torch.nn.functional as F

torch.manual_seed(42)
device = 'cuda'

N = 100
B = 2
C_in, C_out = 64, 64
H, W = 56, 56
K = 3
padding = 1

print("=" * 60)
print("CHANNELS_LAST ACCURACY INVESTIGATION")
print("=" * 60)
print(f"N={N}, B={B}, C={C_in}→{C_out}")

# Create weights and inputs
weights = [torch.randn(C_out, C_in, K, K, device=device) for _ in range(N)]
biases = [torch.randn(C_out, device=device) for _ in range(N)]
x_list = [torch.randn(B, C_in, H, W, device=device) for _ in range(N)]

# Reference: sequential F.conv2d
ref_outputs = [F.conv2d(x_list[i], weights[i], biases[i], padding=padding) for i in range(N)]

# Grouped conv setup
grouped_w = torch.cat(weights, dim=0)
grouped_b = torch.cat(biases, dim=0)
x_packed = torch.stack(x_list, dim=1).view(B, N * C_in, H, W)

# Test 1: Grouped NCHW (original)
print("\n--- Test 1: Grouped NCHW (no format conversion) ---")
out_nchw = F.conv2d(x_packed, grouped_w, grouped_b, padding=padding, groups=N)
out_nchw_list = out_nchw.view(B, N, C_out, H, W).unbind(dim=1)
max_diff_nchw = max((ref_outputs[i] - out_nchw_list[i]).abs().max().item() for i in range(N))
print(f"Max diff: {max_diff_nchw:.6e}")

# Test 2: Grouped NHWC (channels_last)
print("\n--- Test 2: Grouped NHWC (channels_last) ---")
x_nhwc = x_packed.to(memory_format=torch.channels_last)
w_nhwc = grouped_w.to(memory_format=torch.channels_last)
out_nhwc = F.conv2d(x_nhwc, w_nhwc, grouped_b, padding=padding, groups=N)
out_nhwc_cont = out_nhwc.contiguous()
out_nhwc_list = out_nhwc_cont.view(B, N, C_out, H, W).unbind(dim=1)
max_diff_nhwc = max((ref_outputs[i] - out_nhwc_list[i]).abs().max().item() for i in range(N))
print(f"Max diff: {max_diff_nhwc:.6e}")

# Test 3: NCHW vs NHWC directly
print("\n--- Test 3: NCHW vs NHWC (same grouped conv) ---")
diff_nchw_nhwc = (out_nchw - out_nhwc_cont).abs().max().item()
print(f"NCHW vs NHWC diff: {diff_nchw_nhwc:.6e}")

# Test 4: Check if it's the memory format conversion or the kernel
print("\n--- Test 4: Round-trip memory format ---")
x_roundtrip = x_packed.to(memory_format=torch.channels_last).contiguous()
diff_roundtrip = (x_packed - x_roundtrip).abs().max().item()
print(f"Input round-trip diff: {diff_roundtrip:.6e}")

# Test 5: Single conv NCHW vs NHWC
print("\n--- Test 5: Single conv NCHW vs NHWC ---")
single_x = x_list[0]
single_w = weights[0]
single_b = biases[0]

out_single_nchw = F.conv2d(single_x, single_w, single_b, padding=padding)
out_single_nhwc = F.conv2d(
    single_x.to(memory_format=torch.channels_last),
    single_w.to(memory_format=torch.channels_last),
    single_b,
    padding=padding
).contiguous()

diff_single = (out_single_nchw - out_single_nhwc).abs().max().item()
print(f"Single conv NCHW vs NHWC: {diff_single:.6e}")

# Test 6: Check relative error
print("\n--- Test 6: Relative error ---")
ref_max = max(ref_outputs[i].abs().max().item() for i in range(N))
print(f"Reference max value: {ref_max:.2f}")
print(f"NCHW relative error: {max_diff_nchw / ref_max:.6e}")
print(f"NHWC relative error: {max_diff_nhwc / ref_max:.6e}")

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
if max_diff_nchw < 1e-5 and max_diff_nhwc > 1e-4:
    print("The error is from NHWC memory format (different cuDNN kernel).")
    print("NCHW grouped is exact but slow on RTX.")
    print("NHWC grouped is 7x faster but has ~1e-3 absolute error.")
elif max_diff_nchw > 1e-4:
    print("Grouped conv itself has numerical differences (not channels_last).")