"""
Flow Matching Mixture of Experts (FM-MoE) Module

This module implements a Mixture of Experts layer where each expert is a flow matching
model instead of a traditional MLP. Flow matching learns continuous transformations
through normalizing flows, providing more expressive representations than standard
feed-forward networks.

Author: Claude
Date: 2026-02-03
"""

import math
from typing import Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FlowMatchingMoEConfig:
    """
    Configuration class for Flow Matching Mixture of Experts.

    Attributes:
        input_dim: Dimensionality of input features
        hidden_dim: Hidden dimension for flow transformations
        num_experts: Number of flow matching experts in the mixture
        num_selected: Number of top-k experts to activate per token (typically 1-2)
        flow_steps: Number of integration steps for flow matching (higher = more accurate)
        time_embed_dim: Dimension of time embedding for flow conditioning
        use_router_aux_loss: Whether to use auxiliary load balancing loss
        router_z_loss_coef: Coefficient for router z-loss (prevents large logits)
        load_balance_loss_coef: Coefficient for load balancing loss
        expert_capacity_factor: Multiplier for expert capacity limit (1.0 = balanced)
        dropout: Dropout probability for regularization
    """

    input_dim: int
    hidden_dim: int
    num_experts: int = 8
    num_selected: int = 2
    flow_steps: int = 10
    time_embed_dim: int = 64
    use_router_aux_loss: bool = True
    router_z_loss_coef: float = 0.001
    load_balance_loss_coef: float = 0.01
    expert_capacity_factor: float = 1.25
    dropout: float = 0.0


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for flow matching.

    Encodes continuous time values into high-dimensional representations using
    sinusoidal functions at different frequencies, similar to positional encodings
    in transformers.

    Args:
        embed_dim: Dimension of the time embedding
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Embed time values into sinusoidal representations.

        Args:
            time: Time values in [0, 1], shape (batch_size,)

        Returns:
            Time embeddings, shape (batch_size, embed_dim)
        """
        # Ensure time is 1D
        if time.dim() == 0:
            time = time.unsqueeze(0)

        half_dim = self.embed_dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        return embeddings


class FlowMatchingExpert(nn.Module):
    """
    Single flow matching expert network.

    Implements a vector field network that predicts velocities for continuous
    normalizing flows. The network takes an input and time, and outputs a
    velocity field that guides the transformation.

    Args:
        input_dim: Dimension of input features
        hidden_dim: Hidden layer dimension
        time_embed_dim: Dimension of time embeddings
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        # Vector field network (velocity predictor)
        self.network = nn.Sequential(
            nn.Linear(input_dim + time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

        # Initialize last layer to near zero for identity initialization
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute velocity field at position x and time t.

        Args:
            x: Input features, shape (batch_size, input_dim)
            t: Time values in [0, 1], shape (batch_size,)

        Returns:
            Velocity field, shape (batch_size, input_dim)
        """
        # Embed time
        t_emb = self.time_embed(t)

        # Concatenate input with time embedding
        h = torch.cat([x, t_emb], dim=-1)

        # Compute velocity field
        velocity = self.network(h)

        return velocity

    def flow_transform(self, x: torch.Tensor, steps: int = 10) -> torch.Tensor:
        """
        Apply flow transformation via Euler integration.

        Integrates the ODE dx/dt = v(x, t) from t=0 to t=1 using Euler method.

        Args:
            x: Input features, shape (batch_size, input_dim)
            steps: Number of integration steps

        Returns:
            Transformed features, shape (batch_size, input_dim)
        """
        batch_size = x.shape[0]
        device = x.device

        x_t = x
        dt = 1.0 / steps

        for step in range(steps):
            t = step * dt
            t_batch = torch.full((batch_size,), t, device=device, dtype=x.dtype)

            # Compute velocity at current point
            v_t = self.forward(x_t, t_batch)

            # Euler step
            x_t = x_t + v_t * dt

        return x_t


class Router(nn.Module):
    """
    Router network for expert selection.

    Computes routing probabilities for each expert and selects top-k experts
    for each input token. Implements load balancing and stability losses.

    Args:
        input_dim: Dimension of input features
        num_experts: Number of experts to route to
        num_selected: Number of top experts to select per token
    """

    def __init__(self, input_dim: int, num_experts: int, num_selected: int):
        super().__init__()
        self.num_experts = num_experts
        self.num_selected = num_selected

        # Router linear layer
        self.gate = nn.Linear(input_dim, num_experts, bias=False)

        # Initialize with small weights to encourage exploration
        nn.init.normal_(self.gate.weight, std=0.01)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute routing decisions.

        Args:
            x: Input features, shape (batch_size, seq_len, input_dim)

        Returns:
            Tuple of:
                - top_k_probs: Normalized probabilities for selected experts,
                  shape (batch_size, seq_len, num_selected)
                - top_k_indices: Indices of selected experts,
                  shape (batch_size, seq_len, num_selected)
                - all_probs: Probabilities for all experts (for loss computation),
                  shape (batch_size, seq_len, num_experts)
        """
        # Compute routing logits
        logits = self.gate(x)  # (batch_size, seq_len, num_experts)

        # Convert to probabilities
        all_probs = F.softmax(logits, dim=-1)

        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(all_probs, self.num_selected, dim=-1)

        # Normalize selected probabilities
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)

        return top_k_probs, top_k_indices, all_probs

    def compute_router_z_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute router z-loss to encourage smaller logits.

        This loss penalizes large logit values, improving training stability.

        Args:
            logits: Router logits, shape (batch_size, seq_len, num_experts)

        Returns:
            Scalar z-loss value
        """
        log_z = torch.logsumexp(logits, dim=-1)
        z_loss = torch.mean(log_z**2)
        return z_loss

    def compute_load_balance_loss(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing loss to encourage uniform expert usage.

        This loss encourages all experts to be used equally, preventing
        some experts from being consistently ignored.

        Args:
            probs: Expert probabilities, shape (batch_size, seq_len, num_experts)

        Returns:
            Scalar load balance loss
        """
        # Average probability each expert is selected
        mean_probs = probs.mean(dim=[0, 1])  # (num_experts,)

        # Coefficient of variation penalty
        # Perfect balance would have all experts at 1/num_experts
        target = 1.0 / self.num_experts
        loss = torch.sum((mean_probs - target) ** 2)

        return loss * self.num_experts


class FlowMatchingMoE(nn.Module):
    """
    Flow Matching Mixture of Experts Layer.

    A Mixture of Experts layer where each expert is a flow matching network
    that learns continuous transformations. The router selects top-k experts
    for each token and combines their outputs.

    This provides more expressive transformations than traditional MLP-based
    MoE layers through the use of continuous normalizing flows.

    Args:
        config: Configuration object with all hyperparameters

    Example:
        >>> config = FlowMatchingMoEConfig(
        ...     input_dim=512,
        ...     hidden_dim=2048,
        ...     num_experts=8,
        ...     num_selected=2,
        ...     flow_steps=10
        ... )
        >>> moe = FlowMatchingMoE(config)
        >>> x = torch.randn(4, 128, 512)  # (batch, seq_len, input_dim)
        >>> output, aux_loss = moe(x)
        >>> print(output.shape)  # (4, 128, 512)
    """

    def __init__(self, config: FlowMatchingMoEConfig):
        super().__init__()
        self.config = config

        # Router network
        self.router = Router(
            input_dim=config.input_dim,
            num_experts=config.num_experts,
            num_selected=config.num_selected,
        )

        # Flow matching experts
        self.experts = nn.ModuleList(
            [
                FlowMatchingExpert(
                    input_dim=config.input_dim,
                    hidden_dim=config.hidden_dim,
                    time_embed_dim=config.time_embed_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.num_experts)
            ]
        )

    def forward(
        self, x: torch.Tensor, return_aux_loss: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the Flow Matching MoE layer.

        Routes each token to top-k experts, applies flow transformations,
        and combines the results. Optionally computes auxiliary losses for
        load balancing and routing stability.

        Args:
            x: Input features, shape (batch_size, seq_len, input_dim)
            return_aux_loss: Whether to compute and return auxiliary losses

        Returns:
            Tuple of:
                - output: Transformed features, shape (batch_size, seq_len, input_dim)
                - aux_loss: Auxiliary loss (load balance + z-loss) or None
        """
        batch_size, seq_len, input_dim = x.shape

        # Step 1: Route to experts
        top_k_probs, top_k_indices, all_probs = self.router(x)

        # Store logits for z-loss
        logits = self.router.gate(x)

        # Step 2: Initialize output
        output = torch.zeros_like(x)

        # Reshape for easier processing
        x_flat = x.view(-1, input_dim)  # (batch_size * seq_len, input_dim)
        top_k_probs_flat = top_k_probs.view(-1, self.config.num_selected)
        top_k_indices_flat = top_k_indices.view(-1, self.config.num_selected)
        output_flat = output.view(-1, input_dim)

        # Step 3: Process each expert
        for expert_id in range(self.config.num_experts):
            # Find all positions where this expert is selected
            expert_mask = (
                top_k_indices_flat == expert_id
            )  # (batch_size * seq_len, num_selected)

            # Check if any tokens are routed to this expert
            if not expert_mask.any():
                continue

            # Get the positions and selection indices where this expert is used
            positions, selection_indices = torch.where(expert_mask)

            if len(positions) == 0:
                continue

            # Extract inputs for this expert
            expert_inputs = x_flat[positions]  # (num_tokens, input_dim)

            # Get routing weights for this expert
            expert_weights = top_k_probs_flat[
                positions, selection_indices
            ]  # (num_tokens,)

            # Apply flow transformation
            expert_outputs = self.experts[expert_id].flow_transform(
                expert_inputs, steps=self.config.flow_steps
            )

            # Weight the outputs
            weighted_outputs = expert_outputs * expert_weights.unsqueeze(-1)

            # Accumulate to output (scatter-add)
            output_flat.index_add_(0, positions, weighted_outputs)

        # Reshape back to original shape
        output = output_flat.view(batch_size, seq_len, input_dim)

        # Step 4: Compute auxiliary losses
        aux_loss = None
        if return_aux_loss and self.config.use_router_aux_loss and self.training:
            # Load balancing loss
            load_balance_loss = self.router.compute_load_balance_loss(all_probs)

            # Router z-loss
            z_loss = self.router.compute_router_z_loss(logits)

            # Combine losses
            aux_loss = (
                self.config.load_balance_loss_coef * load_balance_loss
                + self.config.router_z_loss_coef * z_loss
            )

        return output, aux_loss

    def get_expert_usage_stats(self, x: torch.Tensor) -> dict:
        """
        Get statistics about expert usage for analysis.

        Args:
            x: Input features, shape (batch_size, seq_len, input_dim)

        Returns:
            Dictionary with expert usage statistics:
                - expert_probs: Mean probability for each expert
                - expert_selections: Number of times each expert was selected
                - balance_score: How balanced the expert usage is (1.0 = perfect)
        """
        with torch.no_grad():
            _, top_k_indices, all_probs = self.router(x)

            # Count selections per expert
            selections = torch.zeros(self.config.num_experts, device=x.device)
            for expert_id in range(self.config.num_experts):
                selections[expert_id] = (top_k_indices == expert_id).sum().float()

            # Mean probabilities
            mean_probs = all_probs.mean(dim=[0, 1])

            # Balance score (1.0 = perfect balance, 0.0 = all on one expert)
            ideal_selection = selections.sum() / self.config.num_experts
            balance_score = 1.0 - (selections - ideal_selection).abs().sum() / (
                2 * selections.sum()
            )

            return {
                "expert_probs": mean_probs.cpu().numpy(),
                "expert_selections": selections.cpu().numpy(),
                "balance_score": balance_score.item(),
            }
