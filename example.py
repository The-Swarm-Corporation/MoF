import torch

from mof.main import FlowMatchingMoE, FlowMatchingMoEConfig

# ============================================================================
# Example Usage and Testing
# ============================================================================


def test_flow_matching_moe():
    """
    Test function demonstrating usage of FlowMatchingMoE.
    """
    print("Testing Flow Matching MoE Module")
    print("=" * 80)

    # Create configuration
    config = FlowMatchingMoEConfig(
        input_dim=512,
        hidden_dim=2048,
        num_experts=8,
        num_selected=2,
        flow_steps=10,
        use_router_aux_loss=True,
    )

    # Create model
    model = FlowMatchingMoE(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print("\nModel Configuration:")
    print(f"  Input dim: {config.input_dim}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Num experts: {config.num_experts}")
    print(f"  Num selected: {config.num_selected}")
    print(f"  Flow steps: {config.flow_steps}")
    print(f"  Total parameters: {total_params:,}")

    # Create sample input
    batch_size, seq_len = 4, 128
    x = torch.randn(batch_size, seq_len, config.input_dim)

    print(f"\nInput shape: {x.shape}")

    out = model(x)

    print(out[0])
    print(out[1])


if __name__ == "__main__":
    test_flow_matching_moe()
