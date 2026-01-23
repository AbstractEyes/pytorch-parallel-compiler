"""
Smoke tst for WideTinyFlux model.

Run locally with:
    python tst_wide_tiny_flux.py
"""

import torch
from models.wide_tiny_flux import WideTinyFlux


def tst_model_init():
    """tst WideTinyFlux can be initialized."""
    print("=" * 80)
    print("tst 1: Model Initialization")
    print("=" * 80)

    n = 4
    hidden_size = 256
    num_heads = 4
    head_dim = 64

    model = WideTinyFlux(
        n=n,
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        in_channels=16,
        joint_attention_dim=768,
        pooled_projection_dim=768,
        num_double_layers=2,
        num_single_layers=2,
        mlp_ratio=4.0,
        guidance_embeds=True,
        strategy='fused',
    )

    print(f"[OK] Model created: {model}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    return model


def tst_forward_pass():
    """tst forward pass with N-first format inputs."""
    print("\n" + "=" * 80)
    print("tst 2: Forward Pass (CPU)")
    print("=" * 80)

    n = 4
    batch_size = 2
    num_patches = 16
    text_len = 32

    model = WideTinyFlux(
        n=n,
        hidden_size=256,
        num_heads=4,
        head_dim=64,
        in_channels=16,
        joint_attention_dim=768,
        pooled_projection_dim=768,
        num_double_layers=2,
        num_single_layers=2,
        mlp_ratio=4.0,
        guidance_embeds=True,
        strategy='fused',
    )

    # Create N-first format inputs
    hidden_states = torch.randn(n, batch_size, num_patches, 16)
    encoder_hidden_states = torch.randn(n, batch_size, text_len, 768)
    pooled_projections = torch.randn(n, batch_size, 768)
    timestep = torch.rand(n, batch_size) * 1000
    img_ids = torch.randn(batch_size, num_patches, 3)
    guidance = torch.rand(n, batch_size) * 5

    print(f"Input shapes:")
    print(f"  hidden_states:          {list(hidden_states.shape)}")
    print(f"  encoder_hidden_states:  {list(encoder_hidden_states.shape)}")
    print(f"  pooled_projections:     {list(pooled_projections.shape)}")
    print(f"  timestep:               {list(timestep.shape)}")
    print(f"  img_ids:                {list(img_ids.shape)}")
    print(f"  guidance:               {list(guidance.shape)}")

    # Forward pass
    with torch.no_grad():
        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=img_ids,
            guidance=guidance,
        )

    print(f"\n[OK] Forward pass successful")
    print(f"  Output shape: {list(output.shape)}")
    print(f"  Expected:     [{n}, {batch_size}, {num_patches}, 16]")

    assert output.shape == (n, batch_size, num_patches, 16), \
        f"Output shape mismatch: {output.shape} != ({n}, {batch_size}, {num_patches}, 16)"

    print(f"  [OK] Output shape correct!")

    return model, output


def tst_cuda_forward():
    """tst forward pass on CUDA."""
    if not torch.cuda.is_available():
        print("\n" + "=" * 80)
        print("tst 3: CUDA Forward Pass - SKIPPED (no CUDA)")
        print("=" * 80)
        return

    print("\n" + "=" * 80)
    print("tst 3: CUDA Forward Pass")
    print("=" * 80)

    n = 8
    batch_size = 4
    num_patches = 64
    text_len = 128

    model = WideTinyFlux(
        n=n,
        hidden_size=512,
        num_heads=8,
        head_dim=64,
        in_channels=16,
        joint_attention_dim=768,
        pooled_projection_dim=768,
        num_double_layers=3,
        num_single_layers=3,
        mlp_ratio=4.0,
        guidance_embeds=True,
        strategy='fused',
    ).cuda()

    # Create CUDA inputs
    hidden_states = torch.randn(n, batch_size, num_patches, 16, device='cuda')
    encoder_hidden_states = torch.randn(n, batch_size, text_len, 768, device='cuda')
    pooled_projections = torch.randn(n, batch_size, 768, device='cuda')
    timestep = torch.rand(n, batch_size, device='cuda') * 1000
    img_ids = torch.randn(batch_size, num_patches, 3, device='cuda')
    guidance = torch.rand(n, batch_size, device='cuda') * 5

    print(f"Input shapes (CUDA):")
    print(f"  hidden_states:          {list(hidden_states.shape)}")
    print(f"  encoder_hidden_states:  {list(encoder_hidden_states.shape)}")
    print(f"  pooled_projections:     {list(pooled_projections.shape)}")
    print(f"  timestep:               {list(timestep.shape)}")
    print(f"  img_ids:                {list(img_ids.shape)}")
    print(f"  guidance:               {list(guidance.shape)}")

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=timestep,
                img_ids=img_ids,
                guidance=guidance,
            )

    # Timed run
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.no_grad():
        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=img_ids,
            guidance=guidance,
        )
    end.record()

    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)

    print(f"\n[OK] CUDA forward pass successful")
    print(f"  Output shape: {list(output.shape)}")
    print(f"  Time:         {elapsed_ms:.2f} ms")
    print(f"  Device:       {output.device}")

    assert output.shape == (n, batch_size, num_patches, 16)
    assert output.device.type == 'cuda'

    print(f"  [OK] All checks passed!")

    return model, output, elapsed_ms


def tst_multiple_strategies():
    """tst different strategies (fused vs sequential)."""
    print("\n" + "=" * 80)
    print("tst 4: Strategy Comparison")
    print("=" * 80)

    n = 4
    batch_size = 2
    num_patches = 16
    text_len = 32

    # Create inputs
    hidden_states = torch.randn(n, batch_size, num_patches, 16)
    encoder_hidden_states = torch.randn(n, batch_size, text_len, 768)
    pooled_projections = torch.randn(n, batch_size, 768)
    timestep = torch.rand(n, batch_size) * 1000
    img_ids = torch.randn(batch_size, num_patches, 3)
    guidance = torch.rand(n, batch_size) * 5

    results = {}

    for strategy in ['fused', 'sequential']:
        print(f"\ntsting strategy: {strategy}")

        model = WideTinyFlux(
            n=n,
            hidden_size=256,
            num_heads=4,
            head_dim=64,
            in_channels=16,
            joint_attention_dim=768,
            pooled_projection_dim=768,
            num_double_layers=2,
            num_single_layers=2,
            mlp_ratio=4.0,
            guidance_embeds=True,
            strategy=strategy,
        )

        with torch.no_grad():
            output = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=timestep,
                img_ids=img_ids,
                guidance=guidance,
            )

        results[strategy] = output
        print(f"  [OK] {strategy:12s} - Output shape: {list(output.shape)}")

    # Compare outputs (should be similar with random init)
    diff = (results['fused'] - results['sequential']).abs().max()
    print(f"\n[OK] Both strategies executed successfully")
    print(f"  Max difference: {diff:.6f} (expected due to random init)")

    return results


def run_all_tsts():
    """Run all smoke tsts."""
    print("\n" + "=" * 80)
    print("WIDE TINY FLUX SMOKE tst SUITE")
    print("=" * 80 + "\n")

    try:
        # tst 1: Init
        model = tst_model_init()

        # tst 2: CPU Forward
        model, output = tst_forward_pass()

        # tst 3: CUDA Forward
        tst_cuda_forward()

        # tst 4: Strategies
        tst_multiple_strategies()

        print("\n" + "=" * 80)
        print("[OK] ALL tstS PASSED")
        print("=" * 80)
        return True

    except Exception as e:
        print("\n" + "=" * 80)
        print("[FAIL] tst FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = run_all_tsts()
    sys.exit(0 if success else 1)
