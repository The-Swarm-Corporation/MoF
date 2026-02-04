from mof.main import FlowMatchingMoE, FlowMatchingMoEConfig
import torch


def visualize_flow_matching_moe(
    model: FlowMatchingMoE = None,
    x: torch.Tensor = None,
    save_path: str = "flow_matching_moe_visualization.png",
    show_flow_trajectory: bool = True,
    figsize: tuple = (20, 16),
) -> None:
    """
    Visualize the Flow Matching Mixture of Experts architecture and operations.

    Creates a comprehensive visualization showing:
    1. High-level architecture diagram
    2. Router probability distribution across experts
    3. Expert selection heatmap
    4. Flow transformation trajectory (ODE integration path)
    5. Expert usage balance statistics

    Args:
        model: FlowMatchingMoE model instance. If None, creates a default one.
        x: Input tensor of shape (batch_size, seq_len, input_dim). If None, creates random input.
        save_path: Path to save the visualization image.
        show_flow_trajectory: Whether to show the flow transformation trajectory.
        figsize: Figure size tuple (width, height).

    Example:
        >>> config = FlowMatchingMoEConfig(input_dim=256, hidden_dim=512, num_experts=8)
        >>> model = FlowMatchingMoE(config)
        >>> x = torch.randn(2, 32, 256)
        >>> visualize_flow_matching_moe(model, x, save_path="my_moe_viz.png")
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
        import numpy as np
    except ImportError:
        print(
            "Matplotlib is required for visualization. Install with: pip install matplotlib"
        )
        return

    # Create default model and input if not provided
    if model is None:
        config = FlowMatchingMoEConfig(
            input_dim=256,
            hidden_dim=512,
            num_experts=8,
            num_selected=2,
            flow_steps=10,
        )
        model = FlowMatchingMoE(config)

    if x is None:
        x = torch.randn(2, 16, model.config.input_dim)

    model.eval()
    config = model.config

    # Get routing information
    with torch.no_grad():
        top_k_probs, top_k_indices, all_probs = model.router(x)
        stats = model.get_expert_usage_stats(x)

    # Create figure with subplots
    fig = plt.figure(figsize=figsize, facecolor="white")

    # Define grid layout
    gs = fig.add_gridspec(
        3, 3, hspace=0.35, wspace=0.3, height_ratios=[1.2, 1, 1]
    )

    # =========================================================================
    # Panel 1: Architecture Diagram (top, spans all columns)
    # =========================================================================
    ax_arch = fig.add_subplot(gs[0, :])
    ax_arch.set_xlim(0, 100)
    ax_arch.set_ylim(0, 50)
    ax_arch.axis("off")
    ax_arch.set_title(
        "Flow Matching Mixture of Experts (FM-MoE) Architecture",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Colors
    input_color = "#3498db"
    router_color = "#e74c3c"
    expert_color = "#2ecc71"
    output_color = "#f39c12"
    arrow_color = "#34495e"

    # Draw input box
    input_box = FancyBboxPatch(
        (2, 20),
        12,
        10,
        boxstyle="round,pad=0.05",
        facecolor=input_color,
        edgecolor="black",
        linewidth=2,
    )
    ax_arch.add_patch(input_box)
    ax_arch.text(
        8,
        25,
        "Input\nTokens",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="white",
    )
    ax_arch.text(
        8,
        17,
        f"({x.shape[0]}×{x.shape[1]}×{x.shape[2]})",
        ha="center",
        va="center",
        fontsize=8,
        color="gray",
    )

    # Draw router box
    router_box = FancyBboxPatch(
        (22, 20),
        14,
        10,
        boxstyle="round,pad=0.05",
        facecolor=router_color,
        edgecolor="black",
        linewidth=2,
    )
    ax_arch.add_patch(router_box)
    ax_arch.text(
        29,
        25,
        "Router\n(Softmax)",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="white",
    )
    ax_arch.text(
        29,
        17,
        f"Top-{config.num_selected} Selection",
        ha="center",
        va="center",
        fontsize=8,
        color="gray",
    )

    # Draw experts
    expert_start_x = 45
    expert_width = 6
    expert_spacing = 1.5
    num_to_draw = min(config.num_experts, 6)

    for i in range(num_to_draw):
        ex = expert_start_x + i * (expert_width + expert_spacing)
        expert_box = FancyBboxPatch(
            (ex, 20),
            expert_width,
            10,
            boxstyle="round,pad=0.03",
            facecolor=expert_color,
            edgecolor="black",
            linewidth=1.5,
        )
        ax_arch.add_patch(expert_box)
        ax_arch.text(
            ex + expert_width / 2,
            25,
            f"E{i}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
        )

    # Draw ellipsis if more experts
    if config.num_experts > 6:
        ax_arch.text(
            expert_start_x + 6 * (expert_width + expert_spacing) - 2,
            25,
            "...",
            fontsize=14,
            fontweight="bold",
        )

    # Expert label
    ax_arch.text(
        expert_start_x
        + (num_to_draw * (expert_width + expert_spacing)) / 2,
        17,
        f"{config.num_experts} Flow Matching Experts",
        ha="center",
        fontsize=9,
        color="gray",
    )

    # Draw weighted sum box
    sum_x = 88
    sum_box = FancyBboxPatch(
        (sum_x, 20),
        10,
        10,
        boxstyle="round,pad=0.05",
        facecolor=output_color,
        edgecolor="black",
        linewidth=2,
    )
    ax_arch.add_patch(sum_box)
    ax_arch.text(
        sum_x + 5,
        25,
        "Weighted\nSum",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="white",
    )

    # Draw arrows
    # Input -> Router
    ax_arch.annotate(
        "",
        xy=(22, 25),
        xytext=(14, 25),
        arrowprops=dict(arrowstyle="->", color=arrow_color, lw=2),
    )

    # Router -> Experts (fan out)
    for i in range(num_to_draw):
        ex = (
            expert_start_x
            + i * (expert_width + expert_spacing)
            + expert_width / 2
        )
        ax_arch.annotate(
            "",
            xy=(ex, 30),
            xytext=(36, 28),
            arrowprops=dict(
                arrowstyle="->",
                color=arrow_color,
                lw=1.5,
                connectionstyle="arc3,rad=0.1",
            ),
        )

    # Experts -> Sum (fan in)
    for i in range(num_to_draw):
        ex = (
            expert_start_x
            + i * (expert_width + expert_spacing)
            + expert_width / 2
        )
        ax_arch.annotate(
            "",
            xy=(sum_x, 25),
            xytext=(ex + expert_width / 2, 25),
            arrowprops=dict(
                arrowstyle="->",
                color=arrow_color,
                lw=1.5,
                connectionstyle="arc3,rad=-0.1",
            ),
        )

    # Draw flow steps detail box
    flow_detail_box = FancyBboxPatch(
        (45, 35),
        40,
        12,
        boxstyle="round,pad=0.05",
        facecolor="#ecf0f1",
        edgecolor="#bdc3c7",
        linewidth=1,
        linestyle="--",
    )
    ax_arch.add_patch(flow_detail_box)
    ax_arch.text(
        65,
        44,
        "Flow Matching Expert (Detail)",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="#2c3e50",
    )
    ax_arch.text(
        65,
        39,
        f"Euler Integration: x(t+dt) = x(t) + v(x,t)·dt   |   {config.flow_steps} steps from t=0 to t=1",
        ha="center",
        fontsize=9,
        color="#7f8c8d",
    )

    # =========================================================================
    # Panel 2: Router Probability Distribution
    # =========================================================================
    ax_probs = fig.add_subplot(gs[1, 0])
    expert_probs = stats["expert_probs"]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, config.num_experts))

    bars = ax_probs.bar(
        range(config.num_experts),
        expert_probs,
        color=colors,
        edgecolor="black",
        linewidth=1,
    )
    ax_probs.axhline(
        y=1.0 / config.num_experts,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Ideal (uniform)",
    )
    ax_probs.set_xlabel("Expert ID", fontsize=11)
    ax_probs.set_ylabel("Mean Routing Probability", fontsize=11)
    ax_probs.set_title(
        "Router Probability Distribution",
        fontsize=12,
        fontweight="bold",
    )
    ax_probs.set_xticks(range(config.num_experts))
    ax_probs.legend(loc="upper right", fontsize=9)
    ax_probs.set_ylim(0, max(expert_probs) * 1.3)

    # Add value labels on bars
    for bar, prob in zip(bars, expert_probs):
        ax_probs.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{prob:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # =========================================================================
    # Panel 3: Expert Selection Heatmap
    # =========================================================================
    ax_heatmap = fig.add_subplot(gs[1, 1])

    # Create selection matrix (show first batch, limited tokens)
    max_tokens = min(x.shape[1], 24)
    selection_matrix = np.zeros((max_tokens, config.num_experts))

    for token_idx in range(max_tokens):
        for k in range(config.num_selected):
            expert_idx = top_k_indices[0, token_idx, k].item()
            prob = top_k_probs[0, token_idx, k].item()
            selection_matrix[token_idx, expert_idx] = prob

    im = ax_heatmap.imshow(
        selection_matrix.T,
        aspect="auto",
        cmap="YlOrRd",
        interpolation="nearest",
    )
    ax_heatmap.set_xlabel("Token Position", fontsize=11)
    ax_heatmap.set_ylabel("Expert ID", fontsize=11)
    ax_heatmap.set_title(
        f"Expert Selection Weights (Batch 0, Top-{config.num_selected})",
        fontsize=12,
        fontweight="bold",
    )
    ax_heatmap.set_yticks(range(config.num_experts))
    ax_heatmap.set_xticks(
        range(0, max_tokens, max(1, max_tokens // 8))
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_heatmap, shrink=0.8)
    cbar.set_label("Selection Weight", fontsize=10)

    # =========================================================================
    # Panel 4: Expert Usage Balance
    # =========================================================================
    ax_usage = fig.add_subplot(gs[1, 2])

    expert_selections = stats["expert_selections"]
    balance_score = stats["balance_score"]

    # Create pie chart for expert usage
    wedges, texts, autotexts = ax_usage.pie(
        expert_selections,
        labels=[f"E{i}" for i in range(config.num_experts)],
        autopct="%1.1f%%",
        colors=colors,
        explode=[0.02] * config.num_experts,
        startangle=90,
    )
    ax_usage.set_title(
        f"Expert Usage Distribution\nBalance Score: {balance_score:.3f}",
        fontsize=12,
        fontweight="bold",
    )

    # =========================================================================
    # Panel 5: Flow Transformation Trajectory
    # =========================================================================
    if show_flow_trajectory:
        ax_flow = fig.add_subplot(gs[2, 0:2])

        # Take a sample input and trace the flow trajectory
        # Use full input_dim for the expert, then project to 2D for visualization
        sample_input = x[0, 0, :].unsqueeze(
            0
        )  # Full input: (1, input_dim)

        # Get trajectory through one expert
        expert_idx = 0
        expert = model.experts[expert_idx]

        # Trace the trajectory
        steps = config.flow_steps
        # Store first 2 dims for 2D visualization
        trajectory = [sample_input[:, :2].detach().numpy()[0]]
        x_t = sample_input.clone()
        dt = 1.0 / steps

        with torch.no_grad():
            for step in range(steps):
                t = step * dt
                t_batch = torch.full((1,), t, dtype=x.dtype)
                v_t = expert.forward(x_t, t_batch)
                x_t = x_t + v_t * dt  # Update full vector
                # Store first 2 dims for visualization
                trajectory.append(x_t[:, :2].detach().numpy()[0])

        trajectory = np.array(trajectory)

        # Plot 2D projection of trajectory
        time_points = np.linspace(0, 1, steps + 1)

        # Create scatter plot with color gradient for time
        scatter = ax_flow.scatter(
            trajectory[:, 0],
            trajectory[:, 1],
            c=time_points,
            cmap="plasma",
            s=100,
            edgecolors="black",
            linewidth=1,
            zorder=3,
        )

        # Draw arrows between consecutive points
        for i in range(len(trajectory) - 1):
            ax_flow.annotate(
                "",
                xy=(trajectory[i + 1, 0], trajectory[i + 1, 1]),
                xytext=(trajectory[i, 0], trajectory[i, 1]),
                arrowprops=dict(
                    arrowstyle="->", color="gray", lw=1.5
                ),
                zorder=2,
            )

        # Mark start and end
        ax_flow.scatter(
            [trajectory[0, 0]],
            [trajectory[0, 1]],
            color="green",
            s=200,
            marker="o",
            label="Start (t=0)",
            edgecolors="black",
            linewidth=2,
            zorder=4,
        )
        ax_flow.scatter(
            [trajectory[-1, 0]],
            [trajectory[-1, 1]],
            color="red",
            s=200,
            marker="s",
            label="End (t=1)",
            edgecolors="black",
            linewidth=2,
            zorder=4,
        )

        cbar_flow = plt.colorbar(scatter, ax=ax_flow, shrink=0.8)
        cbar_flow.set_label("Time t", fontsize=10)

        ax_flow.set_xlabel("Dimension 0", fontsize=11)
        ax_flow.set_ylabel("Dimension 1", fontsize=11)
        ax_flow.set_title(
            f"Flow Transformation Trajectory (Expert 0)\n"
            f"Euler Integration: {steps} steps, dt={dt:.3f}",
            fontsize=12,
            fontweight="bold",
        )
        ax_flow.legend(loc="upper right", fontsize=9)
        ax_flow.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 6: Summary Statistics
    # =========================================================================
    ax_stats = fig.add_subplot(gs[2, 2])
    ax_stats.axis("off")

    # Create text summary
    stats_text = f"""
    ╔══════════════════════════════════════╗
    ║     FM-MoE Configuration Summary     ║
    ╠══════════════════════════════════════╣
    ║  Input Dimension:     {config.input_dim:>12}  ║
    ║  Hidden Dimension:    {config.hidden_dim:>12}  ║
    ║  Number of Experts:   {config.num_experts:>12}  ║
    ║  Top-k Selected:      {config.num_selected:>12}  ║
    ║  Flow Steps:          {config.flow_steps:>12}  ║
    ║  Time Embed Dim:      {config.time_embed_dim:>12}  ║
    ║  Dropout:             {config.dropout:>12.2f}  ║
    ╠══════════════════════════════════════╣
    ║         Input/Output Stats           ║
    ╠══════════════════════════════════════╣
    ║  Batch Size:          {x.shape[0]:>12}  ║
    ║  Sequence Length:     {x.shape[1]:>12}  ║
    ║  Total Tokens:        {x.shape[0] * x.shape[1]:>12}  ║
    ╠══════════════════════════════════════╣
    ║         Expert Usage Stats           ║
    ╠══════════════════════════════════════╣
    ║  Balance Score:       {balance_score:>12.4f}  ║
    ║  Most Used Expert:    {np.argmax(expert_selections):>12}  ║
    ║  Least Used Expert:   {np.argmin(expert_selections):>12}  ║
    ╚══════════════════════════════════════╝
    """

    ax_stats.text(
        0.5,
        0.5,
        stats_text,
        transform=ax_stats.transAxes,
        fontsize=10,
        fontfamily="monospace",
        verticalalignment="center",
        horizontalalignment="center",
        bbox=dict(
            boxstyle="round",
            facecolor="#f8f9fa",
            edgecolor="#dee2e6",
            linewidth=2,
        ),
    )

    # Add overall title
    fig.suptitle("", fontsize=1)  # Placeholder for spacing

    # Save figure
    plt.savefig(
        save_path, dpi=150, bbox_inches="tight", facecolor="white"
    )
    plt.close()

    print("\n" + "=" * 60)
    print(f"Visualization saved to: {save_path}")
    print("=" * 60)
    print("\nVisualization includes:")
    print(
        "  1. Architecture diagram showing Router → Experts → Weighted Sum"
    )
    print(
        f"  2. Router probability distribution across {config.num_experts} experts"
    )
    print("  3. Expert selection heatmap for token routing")
    print("  4. Expert usage pie chart with balance score")
    if show_flow_trajectory:
        print(
            f"  5. Flow transformation trajectory ({config.flow_steps} Euler steps)"
        )
    print("  6. Configuration and statistics summary")
    print()


if __name__ == "__main__":

    visualize_flow_matching_moe()
