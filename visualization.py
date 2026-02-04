from mof.main import FlowMatchingMoE, FlowMatchingMoEConfig
import torch
import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime


# =============================================================================
# METRICS AND EVALUATION DATA STRUCTURES
# =============================================================================


@dataclass
class PerformanceMetrics:
    """Performance metrics for the FM-MoE model."""

    forward_pass_time_ms: float
    backward_pass_time_ms: float
    throughput_tokens_per_sec: float
    throughput_samples_per_sec: float
    peak_memory_mb: float
    allocated_memory_mb: float


@dataclass
class ModelMetrics:
    """Model architecture metrics."""

    total_parameters: int
    trainable_parameters: int
    expert_parameters: int
    router_parameters: int
    parameters_per_expert: int
    model_size_mb: float
    flops_estimate: int


@dataclass
class RoutingMetrics:
    """Router and expert utilization metrics."""

    expert_probabilities: List[float]
    expert_selection_counts: List[int]
    balance_score: float
    routing_entropy: float
    load_imbalance_ratio: float
    expert_utilization_std: float
    router_confidence_mean: float
    router_confidence_std: float
    sparsity_ratio: float


@dataclass
class FlowQualityMetrics:
    """Flow matching quality metrics."""

    flow_magnitude_mean: float
    flow_magnitude_std: float
    transformation_norm_mean: float
    transformation_norm_std: float
    lipschitz_estimate: float
    step_convergence: List[float]
    reconstruction_error: float
    information_preservation_ratio: float


@dataclass
class ScalabilityMetrics:
    """Scalability analysis metrics."""

    num_experts: int
    flow_steps: int
    hidden_dim: int
    batch_size: int
    seq_len: int
    time_per_expert_ms: float
    time_per_flow_step_ms: float
    memory_scaling_factor: float


@dataclass
class ComprehensiveEvaluation:
    """Complete evaluation results for paper."""

    timestamp: str
    config_summary: Dict[str, Any]
    performance: PerformanceMetrics
    model: ModelMetrics
    routing: RoutingMetrics
    flow_quality: FlowQualityMetrics
    scalability: ScalabilityMetrics
    ablation_results: Dict[str, Any]


# =============================================================================
# METRICS COMPUTATION FUNCTIONS
# =============================================================================


def compute_performance_metrics(
    model: FlowMatchingMoE,
    x: torch.Tensor,
    num_warmup: int = 3,
    num_iterations: int = 10,
) -> PerformanceMetrics:
    """
    Compute performance metrics including latency and throughput.

    Args:
        model: FlowMatchingMoE model instance
        x: Input tensor of shape (batch_size, seq_len, input_dim)
        num_warmup: Number of warmup iterations
        num_iterations: Number of timed iterations

    Returns:
        PerformanceMetrics dataclass with timing and memory stats
    """
    batch_size, seq_len, _ = x.shape
    total_tokens = batch_size * seq_len

    # Warmup
    model.eval()
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)

    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Forward pass timing
    forward_times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_times.append((time.perf_counter() - start) * 1000)

    avg_forward_ms = sum(forward_times) / len(forward_times)

    # Backward pass timing
    backward_times = []
    model.train()
    for _ in range(num_iterations):
        x_grad = x.clone().requires_grad_(True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        output, aux_loss = model(x_grad)
        loss = output.sum()
        if aux_loss is not None:
            loss = loss + aux_loss
        loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        backward_times.append((time.perf_counter() - start) * 1000)

    avg_backward_ms = sum(backward_times) / len(backward_times)

    # Memory metrics
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        allocated_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
    else:
        peak_memory_mb = 0.0
        allocated_memory_mb = 0.0

    # Throughput
    throughput_tokens = total_tokens / (avg_forward_ms / 1000)
    throughput_samples = batch_size / (avg_forward_ms / 1000)

    return PerformanceMetrics(
        forward_pass_time_ms=avg_forward_ms,
        backward_pass_time_ms=avg_backward_ms,
        throughput_tokens_per_sec=throughput_tokens,
        throughput_samples_per_sec=throughput_samples,
        peak_memory_mb=peak_memory_mb,
        allocated_memory_mb=allocated_memory_mb,
    )


def compute_model_metrics(model: FlowMatchingMoE) -> ModelMetrics:
    """
    Compute model architecture metrics including parameter counts and FLOPs.

    Args:
        model: FlowMatchingMoE model instance

    Returns:
        ModelMetrics dataclass with architecture statistics
    """
    config = model.config

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Router parameters
    router_params = sum(p.numel() for p in model.router.parameters())

    # Expert parameters
    expert_params = sum(p.numel() for p in model.experts.parameters())
    params_per_expert = expert_params // config.num_experts

    # Model size in MB (assuming float32)
    model_size_mb = total_params * 4 / 1024 / 1024

    # FLOPs estimation (per forward pass)
    # Router: input_dim * num_experts
    router_flops = config.input_dim * config.num_experts

    # Expert (per token, per expert):
    # - Time embedding: embed_dim operations
    # - Network: 3 linear layers
    #   Layer 1: (input_dim + time_embed_dim) * hidden_dim
    #   Layer 2: hidden_dim * hidden_dim
    #   Layer 3: hidden_dim * input_dim
    expert_flops_per_token = (
        config.time_embed_dim  # Time embedding
        + (config.input_dim + config.time_embed_dim) * config.hidden_dim  # Layer 1
        + config.hidden_dim * config.hidden_dim  # Layer 2
        + config.hidden_dim * config.input_dim  # Layer 3
    )
    # Multiply by flow_steps and num_selected
    expert_flops_per_token *= config.flow_steps * config.num_selected

    # Total FLOPs (assuming batch_size=1, seq_len=1 for per-token estimate)
    flops_estimate = router_flops + expert_flops_per_token

    return ModelMetrics(
        total_parameters=total_params,
        trainable_parameters=trainable_params,
        expert_parameters=expert_params,
        router_parameters=router_params,
        parameters_per_expert=params_per_expert,
        model_size_mb=model_size_mb,
        flops_estimate=flops_estimate,
    )


def compute_routing_metrics(model: FlowMatchingMoE, x: torch.Tensor) -> RoutingMetrics:
    """
    Compute routing and expert utilization metrics.

    Args:
        model: FlowMatchingMoE model instance
        x: Input tensor of shape (batch_size, seq_len, input_dim)

    Returns:
        RoutingMetrics dataclass with routing statistics
    """
    import numpy as np

    model.eval()
    with torch.no_grad():
        top_k_probs, top_k_indices, all_probs = model.router(x)
        stats = model.get_expert_usage_stats(x)

    expert_probs = stats["expert_probs"]
    expert_selections = stats["expert_selections"]
    balance_score = stats["balance_score"]

    # Routing entropy (higher = more uncertain/diverse routing)
    # H = -sum(p * log(p))
    all_probs_np = all_probs.mean(dim=[0, 1]).cpu().numpy()
    routing_entropy = -np.sum(all_probs_np * np.log(all_probs_np + 1e-10))

    # Load imbalance ratio (max/min selection ratio)
    if expert_selections.min() > 0:
        load_imbalance = expert_selections.max() / expert_selections.min()
    else:
        load_imbalance = float("inf")

    # Expert utilization standard deviation
    utilization_std = np.std(expert_selections / expert_selections.sum())

    # Router confidence (mean and std of max probability)
    max_probs = all_probs.max(dim=-1).values
    confidence_mean = max_probs.mean().item()
    confidence_std = max_probs.std().item()

    # Sparsity ratio (fraction of near-zero probabilities)
    sparsity_threshold = 0.01
    sparsity_ratio = (all_probs < sparsity_threshold).float().mean().item()

    return RoutingMetrics(
        expert_probabilities=expert_probs.tolist(),
        expert_selection_counts=expert_selections.astype(int).tolist(),
        balance_score=balance_score,
        routing_entropy=routing_entropy,
        load_imbalance_ratio=load_imbalance,
        expert_utilization_std=utilization_std,
        router_confidence_mean=confidence_mean,
        router_confidence_std=confidence_std,
        sparsity_ratio=sparsity_ratio,
    )


def compute_flow_quality_metrics(
    model: FlowMatchingMoE, x: torch.Tensor
) -> FlowQualityMetrics:
    """
    Compute flow matching quality metrics.

    Args:
        model: FlowMatchingMoE model instance
        x: Input tensor of shape (batch_size, seq_len, input_dim)

    Returns:
        FlowQualityMetrics dataclass with flow analysis
    """
    import numpy as np

    config = model.config
    model.eval()

    # Sample input for flow analysis
    sample_input = x[0, 0, :].unsqueeze(0)  # (1, input_dim)
    expert = model.experts[0]

    # Track flow trajectory and velocities
    steps = config.flow_steps
    dt = 1.0 / steps

    velocities = []
    positions = [sample_input.clone()]
    step_changes = []

    x_t = sample_input.clone()

    with torch.no_grad():
        for step in range(steps):
            t = step * dt
            t_batch = torch.full((1,), t, dtype=x.dtype, device=x.device)
            v_t = expert.forward(x_t, t_batch)
            velocities.append(v_t.clone())

            x_prev = x_t.clone()
            x_t = x_t + v_t * dt
            positions.append(x_t.clone())

            # Track per-step change
            step_change = torch.norm(x_t - x_prev).item()
            step_changes.append(step_change)

    # Flow magnitude statistics
    velocity_norms = [torch.norm(v).item() for v in velocities]
    flow_magnitude_mean = np.mean(velocity_norms)
    flow_magnitude_std = np.std(velocity_norms)

    # Transformation statistics
    final_transform = positions[-1] - positions[0]

    # Multi-sample transformation stats
    batch_transforms = [torch.norm(final_transform).item()]  # Include single sample
    for i in range(min(x.shape[0], 4)):
        for j in range(min(x.shape[1], 8)):
            sample = x[i, j, :].unsqueeze(0)
            output = expert.flow_transform(sample, steps=steps)
            batch_transforms.append(torch.norm(output - sample).item())

    transformation_norm_mean = np.mean(batch_transforms)
    transformation_norm_std = np.std(batch_transforms)

    # Lipschitz estimate (local)
    eps = 1e-4
    with torch.no_grad():
        t_mid = torch.tensor([0.5], device=x.device)
        x1 = sample_input
        x2 = sample_input + eps * torch.randn_like(sample_input)
        v1 = expert.forward(x1, t_mid)
        v2 = expert.forward(x2, t_mid)
        lipschitz_estimate = torch.norm(v1 - v2).item() / (
            torch.norm(x1 - x2).item() + 1e-10
        )

    # Step convergence (normalized change per step)
    total_change = sum(step_changes)
    step_convergence = [c / (total_change + 1e-10) for c in step_changes]

    # Reconstruction error (apply flow forward then backward approximation)
    # This tests invertibility of the flow
    x_forward = expert.flow_transform(sample_input, steps=steps)
    # Approximate inverse by integrating backward (not exact for learned flows)
    x_backward = x_forward.clone()
    with torch.no_grad():
        for step in range(steps - 1, -1, -1):
            t = step * dt
            t_batch = torch.full((1,), t, dtype=x.dtype, device=x.device)
            v_t = expert.forward(x_backward, t_batch)
            x_backward = x_backward - v_t * dt  # Reverse Euler

    reconstruction_error = torch.norm(x_backward - sample_input).item()

    # Information preservation (cosine similarity before/after)
    cos_sim = torch.nn.functional.cosine_similarity(
        sample_input, x_forward, dim=-1
    ).item()
    information_preservation = (cos_sim + 1) / 2  # Map from [-1, 1] to [0, 1]

    return FlowQualityMetrics(
        flow_magnitude_mean=flow_magnitude_mean,
        flow_magnitude_std=flow_magnitude_std,
        transformation_norm_mean=transformation_norm_mean,
        transformation_norm_std=transformation_norm_std,
        lipschitz_estimate=lipschitz_estimate,
        step_convergence=step_convergence,
        reconstruction_error=reconstruction_error,
        information_preservation_ratio=information_preservation,
    )


def compute_scalability_metrics(
    model: FlowMatchingMoE,
    x: torch.Tensor,
    num_iterations: int = 5,
) -> ScalabilityMetrics:
    """
    Compute scalability analysis metrics.

    Args:
        model: FlowMatchingMoE model instance
        x: Input tensor of shape (batch_size, seq_len, input_dim)
        num_iterations: Number of iterations for timing

    Returns:
        ScalabilityMetrics dataclass with scaling analysis
    """
    config = model.config
    batch_size, seq_len, _ = x.shape

    model.eval()

    # Time per expert (isolate single expert)
    sample_input = x[0, 0, :].unsqueeze(0)
    expert = model.experts[0]

    expert_times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = expert.flow_transform(sample_input, steps=config.flow_steps)
            expert_times.append((time.perf_counter() - start) * 1000)

    time_per_expert_ms = sum(expert_times) / len(expert_times)

    # Time per flow step
    step_times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            x_t = sample_input.clone()
            dt = 1.0 / config.flow_steps
            start = time.perf_counter()
            for step in range(config.flow_steps):
                t = step * dt
                t_batch = torch.full((1,), t, dtype=x.dtype, device=x.device)
                v_t = expert.forward(x_t, t_batch)
                x_t = x_t + v_t * dt
            step_times.append((time.perf_counter() - start) * 1000)

    time_per_flow_step_ms = (sum(step_times) / len(step_times)) / config.flow_steps

    # Memory scaling factor (ratio of actual to theoretical minimum)
    param_memory = sum(p.numel() * 4 for p in model.parameters()) / 1024 / 1024
    if torch.cuda.is_available():
        actual_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        memory_scaling_factor = actual_memory / (param_memory + 1e-10)
    else:
        memory_scaling_factor = 1.0

    return ScalabilityMetrics(
        num_experts=config.num_experts,
        flow_steps=config.flow_steps,
        hidden_dim=config.hidden_dim,
        batch_size=batch_size,
        seq_len=seq_len,
        time_per_expert_ms=time_per_expert_ms,
        time_per_flow_step_ms=time_per_flow_step_ms,
        memory_scaling_factor=memory_scaling_factor,
    )


def run_ablation_study(
    base_config: FlowMatchingMoEConfig,
    x: torch.Tensor,
    ablation_params: Optional[Dict[str, List]] = None,
) -> Dict[str, Any]:
    """
    Run ablation study varying key hyperparameters.

    Args:
        base_config: Base configuration to modify
        x: Input tensor for evaluation
        ablation_params: Dict of parameter names to lists of values to test

    Returns:
        Dictionary with ablation study results
    """
    if ablation_params is None:
        ablation_params = {
            "num_experts": [2, 4, 8, 16],
            "num_selected": [1, 2, 4],
            "flow_steps": [5, 10, 20, 50],
        }

    results = {}

    for param_name, param_values in ablation_params.items():
        param_results = []

        for value in param_values:
            # Create modified config
            config_dict = {
                "input_dim": base_config.input_dim,
                "hidden_dim": base_config.hidden_dim,
                "num_experts": base_config.num_experts,
                "num_selected": base_config.num_selected,
                "flow_steps": base_config.flow_steps,
                "time_embed_dim": base_config.time_embed_dim,
                "dropout": base_config.dropout,
            }

            # Ensure num_selected doesn't exceed num_experts
            if param_name == "num_experts":
                config_dict[param_name] = value
                config_dict["num_selected"] = min(config_dict["num_selected"], value)
            elif param_name == "num_selected":
                if value <= config_dict["num_experts"]:
                    config_dict[param_name] = value
                else:
                    continue
            else:
                config_dict[param_name] = value

            try:
                test_config = FlowMatchingMoEConfig(**config_dict)
                test_model = FlowMatchingMoE(test_config)

                # Adjust input if needed
                test_x = x
                if test_config.input_dim != x.shape[-1]:
                    test_x = torch.randn(x.shape[0], x.shape[1], test_config.input_dim)

                # Compute metrics
                perf = compute_performance_metrics(
                    test_model, test_x, num_warmup=1, num_iterations=3
                )
                model_metrics = compute_model_metrics(test_model)

                param_results.append(
                    {
                        "value": value,
                        "forward_time_ms": perf.forward_pass_time_ms,
                        "throughput_tokens_per_sec": perf.throughput_tokens_per_sec,
                        "total_parameters": model_metrics.total_parameters,
                        "model_size_mb": model_metrics.model_size_mb,
                    }
                )
            except Exception as e:
                param_results.append(
                    {
                        "value": value,
                        "error": str(e),
                    }
                )

        results[param_name] = param_results

    return results


def run_comprehensive_evaluation(
    model: FlowMatchingMoE,
    x: torch.Tensor,
    run_ablation: bool = True,
) -> ComprehensiveEvaluation:
    """
    Run comprehensive evaluation of the FM-MoE model.

    Args:
        model: FlowMatchingMoE model instance
        x: Input tensor for evaluation
        run_ablation: Whether to run ablation study

    Returns:
        ComprehensiveEvaluation dataclass with all metrics
    """
    config = model.config

    # Config summary
    config_summary = {
        "input_dim": config.input_dim,
        "hidden_dim": config.hidden_dim,
        "num_experts": config.num_experts,
        "num_selected": config.num_selected,
        "flow_steps": config.flow_steps,
        "time_embed_dim": config.time_embed_dim,
        "dropout": config.dropout,
        "use_router_aux_loss": config.use_router_aux_loss,
    }

    # Compute all metrics
    performance = compute_performance_metrics(model, x)
    model_metrics = compute_model_metrics(model)
    routing = compute_routing_metrics(model, x)
    flow_quality = compute_flow_quality_metrics(model, x)
    scalability = compute_scalability_metrics(model, x)

    # Ablation study
    if run_ablation:
        ablation_results = run_ablation_study(config, x)
    else:
        ablation_results = {}

    return ComprehensiveEvaluation(
        timestamp=datetime.now().isoformat(),
        config_summary=config_summary,
        performance=performance,
        model=model_metrics,
        routing=routing,
        flow_quality=flow_quality,
        scalability=scalability,
        ablation_results=ablation_results,
    )


# =============================================================================
# OUTPUT FORMATTING FUNCTIONS
# =============================================================================


def format_metrics_for_paper(evaluation: ComprehensiveEvaluation) -> str:
    """
    Format evaluation results for paper/LaTeX inclusion.

    Args:
        evaluation: ComprehensiveEvaluation dataclass

    Returns:
        Formatted string suitable for paper
    """
    output = []
    output.append("=" * 80)
    output.append("FLOW MATCHING MIXTURE OF EXPERTS (FM-MoE) EVALUATION REPORT")
    output.append("=" * 80)
    output.append(f"Generated: {evaluation.timestamp}")
    output.append("")

    # Configuration
    output.append("-" * 80)
    output.append("MODEL CONFIGURATION")
    output.append("-" * 80)
    for key, value in evaluation.config_summary.items():
        output.append(f"  {key:.<35} {value}")
    output.append("")

    # Model Metrics
    output.append("-" * 80)
    output.append("MODEL ARCHITECTURE METRICS")
    output.append("-" * 80)
    m = evaluation.model
    output.append(f"  Total Parameters:................. {m.total_parameters:,}")
    output.append(f"  Trainable Parameters:............. {m.trainable_parameters:,}")
    output.append(f"  Router Parameters:................ {m.router_parameters:,}")
    output.append(f"  Expert Parameters (total):........ {m.expert_parameters:,}")
    output.append(f"  Parameters per Expert:............ {m.parameters_per_expert:,}")
    output.append(f"  Model Size (MB):.................. {m.model_size_mb:.2f}")
    output.append(f"  FLOPs per Token (estimate):....... {m.flops_estimate:,}")
    output.append("")

    # Performance Metrics
    output.append("-" * 80)
    output.append("PERFORMANCE METRICS")
    output.append("-" * 80)
    p = evaluation.performance
    output.append(f"  Forward Pass Time (ms):........... {p.forward_pass_time_ms:.3f}")
    output.append(f"  Backward Pass Time (ms):.......... {p.backward_pass_time_ms:.3f}")
    output.append(
        f"  Throughput (tokens/sec):.......... {p.throughput_tokens_per_sec:,.0f}"
    )
    output.append(
        f"  Throughput (samples/sec):......... {p.throughput_samples_per_sec:,.1f}"
    )
    output.append(f"  Peak Memory (MB):................. {p.peak_memory_mb:.2f}")
    output.append(f"  Allocated Memory (MB):............ {p.allocated_memory_mb:.2f}")
    output.append("")

    # Routing Metrics
    output.append("-" * 80)
    output.append("ROUTING & EXPERT UTILIZATION METRICS")
    output.append("-" * 80)
    r = evaluation.routing
    output.append(f"  Balance Score:.................... {r.balance_score:.4f}")
    output.append(f"  Routing Entropy:.................. {r.routing_entropy:.4f}")
    output.append(f"  Load Imbalance Ratio:............. {r.load_imbalance_ratio:.4f}")
    output.append(
        f"  Expert Utilization Std:........... {r.expert_utilization_std:.4f}"
    )
    output.append(
        f"  Router Confidence (mean):......... {r.router_confidence_mean:.4f}"
    )
    output.append(f"  Router Confidence (std):.......... {r.router_confidence_std:.4f}")
    output.append(f"  Sparsity Ratio:................... {r.sparsity_ratio:.4f}")
    output.append("")
    output.append("  Expert Probabilities:")
    for i, prob in enumerate(r.expert_probabilities):
        output.append(f"    Expert {i}: {prob:.4f}")
    output.append("")
    output.append("  Expert Selection Counts:")
    for i, count in enumerate(r.expert_selection_counts):
        output.append(f"    Expert {i}: {count}")
    output.append("")

    # Flow Quality Metrics
    output.append("-" * 80)
    output.append("FLOW MATCHING QUALITY METRICS")
    output.append("-" * 80)
    f = evaluation.flow_quality
    output.append(f"  Flow Magnitude (mean):............ {f.flow_magnitude_mean:.4f}")
    output.append(f"  Flow Magnitude (std):............. {f.flow_magnitude_std:.4f}")
    output.append(
        f"  Transformation Norm (mean):....... {f.transformation_norm_mean:.4f}"
    )
    output.append(
        f"  Transformation Norm (std):........ {f.transformation_norm_std:.4f}"
    )
    output.append(f"  Lipschitz Estimate:............... {f.lipschitz_estimate:.4f}")
    output.append(f"  Reconstruction Error:............. {f.reconstruction_error:.4f}")
    output.append(
        f"  Information Preservation:......... {f.information_preservation_ratio:.4f}"
    )
    output.append("")
    output.append("  Step Convergence (normalized):")
    for i, conv in enumerate(f.step_convergence):
        output.append(f"    Step {i}: {conv:.4f}")
    output.append("")

    # Scalability Metrics
    output.append("-" * 80)
    output.append("SCALABILITY METRICS")
    output.append("-" * 80)
    s = evaluation.scalability
    output.append(f"  Batch Size:...................... {s.batch_size}")
    output.append(f"  Sequence Length:................. {s.seq_len}")
    output.append(f"  Time per Expert (ms):............ {s.time_per_expert_ms:.3f}")
    output.append(f"  Time per Flow Step (ms):......... {s.time_per_flow_step_ms:.3f}")
    output.append(f"  Memory Scaling Factor:........... {s.memory_scaling_factor:.2f}")
    output.append("")

    # Ablation Results
    if evaluation.ablation_results:
        output.append("-" * 80)
        output.append("ABLATION STUDY RESULTS")
        output.append("-" * 80)

        for param_name, results in evaluation.ablation_results.items():
            output.append(f"\n  Varying {param_name}:")
            output.append(
                f"  {'Value':>10} {'Time(ms)':>12} {'Throughput':>15} {'Params':>12} {'Size(MB)':>10}"
            )
            output.append("  " + "-" * 65)
            for r in results:
                if "error" in r:
                    output.append(f"  {r['value']:>10} {'ERROR':>12}")
                else:
                    output.append(
                        f"  {r['value']:>10} {r['forward_time_ms']:>12.3f} "
                        f"{r['throughput_tokens_per_sec']:>15,.0f} "
                        f"{r['total_parameters']:>12,} {r['model_size_mb']:>10.2f}"
                    )

    output.append("")
    output.append("=" * 80)
    output.append("END OF EVALUATION REPORT")
    output.append("=" * 80)

    return "\n".join(output)


def generate_latex_table(evaluation: ComprehensiveEvaluation) -> str:
    """
    Generate LaTeX table for paper inclusion.

    Args:
        evaluation: ComprehensiveEvaluation dataclass

    Returns:
        LaTeX formatted table string
    """
    output = []

    # Model metrics table
    output.append("% Model Architecture Metrics")
    output.append("\\begin{table}[h]")
    output.append("\\centering")
    output.append("\\caption{FM-MoE Model Architecture}")
    output.append("\\label{tab:model-architecture}")
    output.append("\\begin{tabular}{lr}")
    output.append("\\toprule")
    output.append("Metric & Value \\\\")
    output.append("\\midrule")

    m = evaluation.model
    output.append(f"Total Parameters & {m.total_parameters:,} \\\\")
    output.append(f"Parameters per Expert & {m.parameters_per_expert:,} \\\\")
    output.append(f"Model Size (MB) & {m.model_size_mb:.2f} \\\\")
    output.append(f"FLOPs per Token & {m.flops_estimate:,} \\\\")

    output.append("\\bottomrule")
    output.append("\\end{tabular}")
    output.append("\\end{table}")
    output.append("")

    # Performance table
    output.append("% Performance Metrics")
    output.append("\\begin{table}[h]")
    output.append("\\centering")
    output.append("\\caption{FM-MoE Performance Benchmarks}")
    output.append("\\label{tab:performance}")
    output.append("\\begin{tabular}{lr}")
    output.append("\\toprule")
    output.append("Metric & Value \\\\")
    output.append("\\midrule")

    p = evaluation.performance
    output.append(f"Forward Pass (ms) & {p.forward_pass_time_ms:.3f} \\\\")
    output.append(f"Backward Pass (ms) & {p.backward_pass_time_ms:.3f} \\\\")
    output.append(f"Throughput (tokens/sec) & {p.throughput_tokens_per_sec:,.0f} \\\\")

    output.append("\\bottomrule")
    output.append("\\end{tabular}")
    output.append("\\end{table}")
    output.append("")

    # Routing metrics table
    output.append("% Routing Metrics")
    output.append("\\begin{table}[h]")
    output.append("\\centering")
    output.append("\\caption{Expert Routing Analysis}")
    output.append("\\label{tab:routing}")
    output.append("\\begin{tabular}{lr}")
    output.append("\\toprule")
    output.append("Metric & Value \\\\")
    output.append("\\midrule")

    r = evaluation.routing
    output.append(f"Balance Score & {r.balance_score:.4f} \\\\")
    output.append(f"Routing Entropy & {r.routing_entropy:.4f} \\\\")
    output.append(f"Load Imbalance Ratio & {r.load_imbalance_ratio:.4f} \\\\")
    output.append(
        f"Router Confidence & {r.router_confidence_mean:.4f} $\\pm$ {r.router_confidence_std:.4f} \\\\"
    )

    output.append("\\bottomrule")
    output.append("\\end{tabular}")
    output.append("\\end{table}")
    output.append("")

    # Flow quality table
    output.append("% Flow Quality Metrics")
    output.append("\\begin{table}[h]")
    output.append("\\centering")
    output.append("\\caption{Flow Matching Quality Analysis}")
    output.append("\\label{tab:flow-quality}")
    output.append("\\begin{tabular}{lr}")
    output.append("\\toprule")
    output.append("Metric & Value \\\\")
    output.append("\\midrule")

    f = evaluation.flow_quality
    output.append(
        f"Flow Magnitude & {f.flow_magnitude_mean:.4f} $\\pm$ {f.flow_magnitude_std:.4f} \\\\"
    )
    output.append(
        f"Transformation Norm & {f.transformation_norm_mean:.4f} $\\pm$ {f.transformation_norm_std:.4f} \\\\"
    )
    output.append(f"Lipschitz Estimate & {f.lipschitz_estimate:.4f} \\\\")
    output.append(f"Reconstruction Error & {f.reconstruction_error:.4f} \\\\")
    output.append(
        f"Information Preservation & {f.information_preservation_ratio:.4f} \\\\"
    )

    output.append("\\bottomrule")
    output.append("\\end{tabular}")
    output.append("\\end{table}")

    return "\n".join(output)


def _convert_to_native_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to native Python types for JSON serialization.

    Args:
        obj: Object to convert (can be dict, list, or scalar)

    Returns:
        Object with all numpy types converted to native Python types
    """
    import numpy as np

    if isinstance(obj, dict):
        return {k: _convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_native_types(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def save_evaluation_results(
    evaluation: ComprehensiveEvaluation,
    output_dir: str = "evals",
    prefix: str = "fm_moe_eval",
) -> Dict[str, str]:
    """
    Save evaluation results to multiple formats.

    Args:
        evaluation: ComprehensiveEvaluation dataclass
        output_dir: Directory to save results
        prefix: Filename prefix

    Returns:
        Dictionary with paths to saved files
    """
    import os

    os.makedirs(output_dir, exist_ok=True)
    saved_files = {}

    # Save JSON
    json_path = os.path.join(output_dir, f"{prefix}_results.json")
    with open(json_path, "w") as f:
        # Convert dataclasses to dicts and ensure native Python types
        data = {
            "timestamp": evaluation.timestamp,
            "config_summary": evaluation.config_summary,
            "performance": asdict(evaluation.performance),
            "model": asdict(evaluation.model),
            "routing": asdict(evaluation.routing),
            "flow_quality": asdict(evaluation.flow_quality),
            "scalability": asdict(evaluation.scalability),
            "ablation_results": evaluation.ablation_results,
        }
        # Convert all numpy types to native Python types
        data = _convert_to_native_types(data)
        json.dump(data, f, indent=2)
    saved_files["json"] = json_path

    # Save text report
    text_path = os.path.join(output_dir, f"{prefix}_report.txt")
    with open(text_path, "w") as f:
        f.write(format_metrics_for_paper(evaluation))
    saved_files["text"] = text_path

    # Save LaTeX
    latex_path = os.path.join(output_dir, f"{prefix}_tables.tex")
    with open(latex_path, "w") as f:
        f.write(generate_latex_table(evaluation))
    saved_files["latex"] = latex_path

    return saved_files


# =============================================================================
# BENCHMARK COMPARISON FUNCTIONS
# =============================================================================


def run_benchmark_suite(
    configs: Optional[List[Dict]] = None,
    batch_sizes: List[int] = [1, 4, 8, 16],
    seq_lengths: List[int] = [32, 64, 128, 256],
) -> Dict[str, Any]:
    """
    Run comprehensive benchmark suite across configurations.

    Args:
        configs: List of config dictionaries to test
        batch_sizes: List of batch sizes to benchmark
        seq_lengths: List of sequence lengths to benchmark

    Returns:
        Dictionary with benchmark results
    """
    if configs is None:
        configs = [
            {"name": "small", "input_dim": 128, "hidden_dim": 256, "num_experts": 4},
            {"name": "medium", "input_dim": 256, "hidden_dim": 512, "num_experts": 8},
            {"name": "large", "input_dim": 512, "hidden_dim": 1024, "num_experts": 16},
        ]

    results = {
        "configs": [],
        "batch_scaling": [],
        "sequence_scaling": [],
    }

    # Test each configuration
    for config_dict in configs:
        name = config_dict.pop("name", "unnamed")
        config = FlowMatchingMoEConfig(**config_dict, num_selected=2, flow_steps=10)
        model = FlowMatchingMoE(config)

        x = torch.randn(4, 64, config.input_dim)

        perf = compute_performance_metrics(model, x, num_warmup=1, num_iterations=3)
        model_metrics = compute_model_metrics(model)

        results["configs"].append(
            {
                "name": name,
                "config": {**config_dict, "name": name},
                "forward_time_ms": perf.forward_pass_time_ms,
                "throughput": perf.throughput_tokens_per_sec,
                "parameters": model_metrics.total_parameters,
                "model_size_mb": model_metrics.model_size_mb,
            }
        )

    # Batch size scaling (use medium config)
    medium_config = FlowMatchingMoEConfig(
        input_dim=256, hidden_dim=512, num_experts=8, num_selected=2, flow_steps=10
    )
    medium_model = FlowMatchingMoE(medium_config)

    for bs in batch_sizes:
        x = torch.randn(bs, 64, 256)
        perf = compute_performance_metrics(
            medium_model, x, num_warmup=1, num_iterations=3
        )
        results["batch_scaling"].append(
            {
                "batch_size": bs,
                "forward_time_ms": perf.forward_pass_time_ms,
                "throughput": perf.throughput_tokens_per_sec,
            }
        )

    # Sequence length scaling
    for sl in seq_lengths:
        x = torch.randn(4, sl, 256)
        perf = compute_performance_metrics(
            medium_model, x, num_warmup=1, num_iterations=3
        )
        results["sequence_scaling"].append(
            {
                "seq_length": sl,
                "forward_time_ms": perf.forward_pass_time_ms,
                "throughput": perf.throughput_tokens_per_sec,
            }
        )

    return results


def print_benchmark_summary(benchmark_results: Dict[str, Any]) -> None:
    """Print formatted benchmark summary."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUITE RESULTS")
    print("=" * 80)

    print("\n--- Configuration Comparison ---")
    print(
        f"{'Config':<12} {'Time(ms)':<12} {'Throughput':<15} {'Params':<12} {'Size(MB)':<10}"
    )
    print("-" * 65)
    for r in benchmark_results["configs"]:
        print(
            f"{r['name']:<12} {r['forward_time_ms']:<12.3f} {r['throughput']:<15,.0f} "
            f"{r['parameters']:<12,} {r['model_size_mb']:<10.2f}"
        )

    print("\n--- Batch Size Scaling ---")
    print(f"{'Batch':<10} {'Time(ms)':<12} {'Throughput':<15}")
    print("-" * 40)
    for r in benchmark_results["batch_scaling"]:
        print(
            f"{r['batch_size']:<10} {r['forward_time_ms']:<12.3f} {r['throughput']:<15,.0f}"
        )

    print("\n--- Sequence Length Scaling ---")
    print(f"{'SeqLen':<10} {'Time(ms)':<12} {'Throughput':<15}")
    print("-" * 40)
    for r in benchmark_results["sequence_scaling"]:
        print(
            f"{r['seq_length']:<10} {r['forward_time_ms']:<12.3f} {r['throughput']:<15,.0f}"
        )

    print("\n" + "=" * 80)


# =============================================================================
# VISUALIZATION FUNCTION
# =============================================================================


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
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3, height_ratios=[1.2, 1, 1])

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
        expert_start_x + (num_to_draw * (expert_width + expert_spacing)) / 2,
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
        ex = expert_start_x + i * (expert_width + expert_spacing) + expert_width / 2
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
        ex = expert_start_x + i * (expert_width + expert_spacing) + expert_width / 2
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
    ax_heatmap.set_xticks(range(0, max_tokens, max(1, max_tokens // 8)))

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
        sample_input = x[0, 0, :].unsqueeze(0)  # Full input: (1, input_dim)

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
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
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
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print("\n" + "=" * 60)
    print(f"Visualization saved to: {save_path}")
    print("=" * 60)
    print("\nVisualization includes:")
    print("  1. Architecture diagram showing Router → Experts → Weighted Sum")
    print(f"  2. Router probability distribution across {config.num_experts} experts")
    print("  3. Expert selection heatmap for token routing")
    print("  4. Expert usage pie chart with balance score")
    if show_flow_trajectory:
        print(f"  5. Flow transformation trajectory ({config.flow_steps} Euler steps)")
    print("  6. Configuration and statistics summary")
    print()


def run_full_evaluation_and_visualization(
    model: FlowMatchingMoE = None,
    x: torch.Tensor = None,
    save_dir: str = "evals",
    run_ablation: bool = True,
    run_benchmarks: bool = True,
    save_results: bool = True,
) -> ComprehensiveEvaluation:
    """
    Run complete evaluation pipeline including visualization, metrics, and benchmarks.

    This is the main entry point for generating paper-ready results.

    Args:
        model: FlowMatchingMoE model instance. If None, creates default.
        x: Input tensor. If None, creates random input.
        save_dir: Directory to save all outputs
        run_ablation: Whether to run ablation study
        run_benchmarks: Whether to run benchmark suite
        save_results: Whether to save results to files

    Returns:
        ComprehensiveEvaluation dataclass with all metrics
    """
    print("\n" + "=" * 80)
    print("FLOW MATCHING MIXTURE OF EXPERTS (FM-MoE)")
    print("Comprehensive Evaluation and Benchmarking Suite")
    print("=" * 80)

    # Create default model and input if not provided
    if model is None:
        print("\n[1/6] Creating default FM-MoE model...")
        config = FlowMatchingMoEConfig(
            input_dim=256,
            hidden_dim=512,
            num_experts=8,
            num_selected=2,
            flow_steps=10,
        )
        model = FlowMatchingMoE(config)
        print(
            f"      Model created with {config.num_experts} experts, "
            f"dim={config.input_dim}, hidden={config.hidden_dim}"
        )
    else:
        print("\n[1/6] Using provided FM-MoE model...")

    if x is None:
        x = torch.randn(4, 32, model.config.input_dim)
        print(f"      Created input tensor: {x.shape}")

    # Run comprehensive evaluation
    print("\n[2/6] Running comprehensive evaluation...")
    evaluation = run_comprehensive_evaluation(model, x, run_ablation=run_ablation)
    print("      Evaluation complete!")

    # Print formatted report
    print("\n[3/6] Generating evaluation report...")
    report = format_metrics_for_paper(evaluation)
    print(report)

    # Run benchmark suite if requested
    if run_benchmarks:
        print("\n[4/6] Running benchmark suite...")
        benchmark_results = run_benchmark_suite()
        print_benchmark_summary(benchmark_results)
    else:
        print("\n[4/6] Skipping benchmark suite...")
        benchmark_results = None

    # Generate visualization
    print("\n[5/6] Generating architecture visualization...")
    import os

    os.makedirs(save_dir, exist_ok=True)
    viz_path = f"{save_dir}/flow_matching_moe_visualization.png"
    visualize_flow_matching_moe(model, x, save_path=viz_path)

    # Save results
    if save_results:
        print("\n[6/6] Saving results to files...")
        saved_files = save_evaluation_results(
            evaluation, output_dir=save_dir, prefix="fm_moe_eval"
        )
        print(f"      JSON results: {saved_files['json']}")
        print(f"      Text report:  {saved_files['text']}")
        print(f"      LaTeX tables: {saved_files['latex']}")

        # Save benchmark results if available
        if benchmark_results:
            import os

            benchmark_path = os.path.join(save_dir, "fm_moe_benchmarks.json")
            with open(benchmark_path, "w") as f:
                json.dump(benchmark_results, f, indent=2)
            print(f"      Benchmarks:   {benchmark_path}")
    else:
        print("\n[6/6] Skipping file save...")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nOutput files saved to '{save_dir}/' folder:")
    print(f"  - {save_dir}/fm_moe_eval_results.json    : Raw metrics data")
    print(f"  - {save_dir}/fm_moe_eval_report.txt      : Formatted text report")
    print(f"  - {save_dir}/fm_moe_eval_tables.tex      : LaTeX tables for paper")
    print(f"  - {save_dir}/fm_moe_benchmarks.json      : Benchmark suite results")
    print(f"  - {save_dir}/flow_matching_moe_visualization.png : Architecture diagram")
    print("\nKey metrics for abstract/introduction:")
    print(f"  - Total Parameters: {evaluation.model.total_parameters:,}")
    print(
        f"  - Throughput: {evaluation.performance.throughput_tokens_per_sec:,.0f} tokens/sec"
    )
    print(f"  - Expert Balance Score: {evaluation.routing.balance_score:.4f}")
    print(
        f"  - Information Preservation: {evaluation.flow_quality.information_preservation_ratio:.4f}"
    )
    print()

    return evaluation


if __name__ == "__main__":
    # Run full evaluation pipeline
    evaluation = run_full_evaluation_and_visualization(
        save_dir="evals",
        run_ablation=True,
        run_benchmarks=True,
        save_results=True,
    )

    # Example: Access specific metrics programmatically
    print("\n--- Quick Access to Key Metrics ---")
    print(f"Forward latency: {evaluation.performance.forward_pass_time_ms:.3f} ms")
    print(f"Model size: {evaluation.model.model_size_mb:.2f} MB")
    print(f"Routing entropy: {evaluation.routing.routing_entropy:.4f}")
    print(f"Flow Lipschitz estimate: {evaluation.flow_quality.lipschitz_estimate:.4f}")
