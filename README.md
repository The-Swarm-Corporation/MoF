# Flow Matching Mixture of Experts (FM-MoE)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

A PyTorch implementation of Flow Matching Mixture of Experts, a novel neural network architecture that integrates continuous normalizing flows with sparse mixture of experts routing for enhanced representational capacity in deep learning models.

---

## Abstract

Mixture of Experts (MoE) architectures have demonstrated remarkable success in scaling neural networks by activating only a subset of parameters per input. However, traditional MoE implementations rely on standard multi-layer perceptrons (MLPs) as expert networks, which may limit the expressiveness of learned transformations. This work introduces **Flow Matching Mixture of Experts (FM-MoE)**, a framework that replaces conventional MLP experts with flow matching networks. Each expert learns a continuous transformation through an ordinary differential equation (ODE), enabling more expressive feature mappings while maintaining the computational efficiency of sparse expert selection.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture](#architecture)
3. [Mathematical Formulation](#mathematical-formulation)
4. [Installation](#installation)
5. [Usage](#usage)
6. [API Reference](#api-reference)
7. [Configuration](#configuration)
8. [Experimental Results](#experimental-results)
9. [Citation](#citation)
10. [License](#license)

---

## Introduction

### Background

**Mixture of Experts (MoE)** models partition the input space among specialized sub-networks (experts), with a gating mechanism that routes each input to a subset of experts. This approach enables conditional computation, where only a fraction of model parameters are activated for any given input, allowing for significant scaling of model capacity without proportional increases in computational cost (Shazeer et al., 2017; Fedus et al., 2022).

**Flow Matching** is a simulation-free approach to training continuous normalizing flows (CNFs). Unlike traditional normalizing flows that require explicit invertibility constraints or simulation during training, flow matching learns a velocity field that transports samples from a source distribution to a target distribution along optimal transport paths (Lipman et al., 2023).

### Motivation

Traditional MoE architectures employ feed-forward networks as experts, which apply fixed, discrete transformations to inputs. By replacing these with flow matching networks, FM-MoE enables each expert to learn a continuous, time-dependent transformation governed by:

```
dx/dt = v_theta(x, t), where t in [0, 1]
```

This formulation provides several theoretical and practical advantages:

1. **Enhanced Expressiveness**: Continuous transformations can represent more complex mappings than discrete layer compositions
2. **Smooth Interpolation**: The ODE formulation ensures smooth transitions in the learned representation space
3. **Flexible Computation**: Integration steps can be adjusted at inference time for accuracy-efficiency trade-offs

---

## Architecture

### High-Level Overview

```
                                    +------------------+
                                    |   Expert 0       |
                                    | (Flow Matching)  |
                                    +--------+---------+
                                             |
+----------+    +----------+    +------------------+    +----------+
|  Input   | -> |  Router  | -> |   Expert 1       | -> | Weighted |
|  Tokens  |    | (Top-k)  |    | (Flow Matching)  |    |   Sum    | -> Output
+----------+    +----------+    +--------+---------+    +----------+
                    |                    |
                    |           +------------------+
                    +---------> |   Expert N-1     |
                                | (Flow Matching)  |
                                +------------------+
```

### Component Architecture

#### 1. Router Network

The router computes expert assignment probabilities and selects the top-k experts for each token:

```
                    +-------------------------+
                    |     Linear(d -> E)      |
      Input  -----> |          |              | -----> Top-k Selection
     (B,L,d)        |     Softmax             |        Probabilities
                    +-------------------------+
```

#### 2. Flow Matching Expert

Each expert implements a velocity field network integrated via Euler method:

```
    +-------+     +---------------+     +-------+
    | x(t)  | --> | Concat with   | --> |  MLP  | --> v(x,t)
    +-------+     | time embed    |     +-------+
                  +---------------+
         |                                   |
         |        x(t+dt) = x(t) + v*dt      |
         +-----------------------------------+
                  (Euler Integration)
```

#### 3. Flow Matching Expert Internal Structure

```
+------------------------------------------------------------------+
|                    Flow Matching Expert                           |
+------------------------------------------------------------------+
|                                                                   |
|   Input (d)                                                       |
|      |                                                            |
|      v                                                            |
|   +------------------+    +----------------------+                |
|   | Sinusoidal Time  |    |                      |                |
|   | Embedding        |--->|  Concatenate         |                |
|   | (t -> d_time)    |    |  [x; time_embed]     |                |
|   +------------------+    +----------+-----------+                |
|                                      |                            |
|                                      v                            |
|                          +-----------------------+                |
|                          | Linear(d+d_time, h)   |                |
|                          +-----------------------+                |
|                                      |                            |
|                                      v                            |
|                          +-----------------------+                |
|                          | SiLU + LayerNorm      |                |
|                          +-----------------------+                |
|                                      |                            |
|                                      v                            |
|                          +-----------------------+                |
|                          | Linear(h, h)          |                |
|                          +-----------------------+                |
|                                      |                            |
|                                      v                            |
|                          +-----------------------+                |
|                          | SiLU + LayerNorm      |                |
|                          +-----------------------+                |
|                                      |                            |
|                                      v                            |
|                          +-----------------------+                |
|                          | Linear(h, d)          |                |
|                          | (zero-initialized)    |                |
|                          +-----------------------+                |
|                                      |                            |
|                                      v                            |
|                              Velocity v(x,t)                      |
|                                                                   |
+------------------------------------------------------------------+
```

---

## Mathematical Formulation

### Flow Matching Fundamentals

Flow matching learns a time-dependent velocity field `v_theta(x, t)` that defines an ordinary differential equation:

```
dx/dt = v_theta(x, t),    t in [0, 1]
```

The transformation from input `x_0` to output `x_1` is obtained by integrating this ODE:

```
x_1 = x_0 + integral from 0 to 1 of v_theta(x(t), t) dt
```

### Numerical Integration

We employ the Euler method for numerical integration with `N` steps:

```
x_{n+1} = x_n + v_theta(x_n, t_n) * dt

where:
  - dt = 1/N (step size)
  - t_n = n * dt (time at step n)
  - n in {0, 1, ..., N-1}
```

### Expert Routing

Given input `x in R^d`, the router computes expert probabilities:

```
p(e|x) = softmax(W_r * x)_e,    e in {1, ..., E}
```

The top-k experts are selected based on these probabilities:

```
S(x) = argmax_k { p(e|x) : e in {1, ..., E} }
```

### Mixture Computation

The final output combines selected expert outputs:

```
y = sum over e in S(x) of [ p_norm(e|x) * f_e(x) ]

where:
  - f_e(x) is the flow transformation of expert e
  - p_norm(e|x) = p(e|x) / sum over e' in S(x) of p(e'|x)
```

### Auxiliary Losses

#### Load Balancing Loss

Encourages uniform expert utilization:

```
L_balance = E * sum from e=1 to E of (p_bar_e - 1/E)^2

where p_bar_e = (1/BL) * sum over b,l of p(e|x_{b,l})
```

#### Router Z-Loss

Prevents large logit values for training stability:

```
L_z = (1/BL) * sum over b,l of (log(sum over e of exp(z_{b,l,e})))^2
```

### Time Embedding

Sinusoidal positional encoding for continuous time:

```
PE(t, 2i) = sin(t / 10000^(2i/d))
PE(t, 2i+1) = cos(t / 10000^(2i/d))
```

---

## Installation

### Requirements

- Python 3.8 or higher
- PyTorch 2.0 or higher

### Install from Source

```bash
git clone https://github.com/The-Swarm-Corporation/MoF.git
cd MoF
pip install -e .
```

### Dependencies

```bash
pip install torch>=2.0.0
pip install matplotlib  # Optional, for visualization
```

---

## Usage

### Basic Example

```python
import torch
from mof import FlowMatchingMoE, FlowMatchingMoEConfig

# Define configuration
config = FlowMatchingMoEConfig(
    input_dim=512,
    hidden_dim=2048,
    num_experts=8,
    num_selected=2,
    flow_steps=10,
)

# Create model
model = FlowMatchingMoE(config)

# Forward pass
x = torch.randn(4, 128, 512)  # (batch_size, seq_len, input_dim)
output, aux_loss = model(x)

print(f"Output shape: {output.shape}")  # (4, 128, 512)
print(f"Auxiliary loss: {aux_loss.item():.6f}")
```

### Integration with Transformer

```python
import torch
import torch.nn as nn
from mof import FlowMatchingMoE, FlowMatchingMoEConfig

class TransformerBlockWithFMMoE(nn.Module):
    """Transformer block using FM-MoE instead of standard FFN."""
    
    def __init__(self, d_model: int, n_heads: int, num_experts: int = 8):
        super().__init__()
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        
        # FM-MoE replaces the feed-forward network
        self.moe = FlowMatchingMoE(FlowMatchingMoEConfig(
            input_dim=d_model,
            hidden_dim=d_model * 4,
            num_experts=num_experts,
            num_selected=2,
            flow_steps=10,
        ))
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # FM-MoE with residual
        moe_out, aux_loss = self.moe(x)
        x = self.norm2(x + moe_out)
        
        return x, aux_loss
```

### Expert Usage Analysis

```python
# Get expert usage statistics
stats = model.get_expert_usage_stats(x)

print(f"Expert probabilities: {stats['expert_probs']}")
print(f"Expert selections: {stats['expert_selections']}")
print(f"Balance score: {stats['balance_score']:.4f}")  # 1.0 = perfect balance
```

### Visualization

```python
from mof.main import visualize_flow_matching_moe

# Generate comprehensive visualization
visualize_flow_matching_moe(
    model=model,
    x=x,
    save_path="fm_moe_analysis.png",
    show_flow_trajectory=True,
)
```

---

## API Reference

### FlowMatchingMoEConfig

Configuration dataclass for FM-MoE hyperparameters.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dim` | int | required | Dimensionality of input features |
| `hidden_dim` | int | required | Hidden dimension for flow transformations |
| `num_experts` | int | 8 | Number of flow matching experts |
| `num_selected` | int | 2 | Number of top-k experts per token |
| `flow_steps` | int | 10 | Euler integration steps |
| `time_embed_dim` | int | 64 | Time embedding dimension |
| `use_router_aux_loss` | bool | True | Enable auxiliary losses |
| `router_z_loss_coef` | float | 0.001 | Z-loss coefficient |
| `load_balance_loss_coef` | float | 0.01 | Load balancing coefficient |
| `expert_capacity_factor` | float | 1.25 | Expert capacity multiplier |
| `dropout` | float | 0.0 | Dropout probability |

### FlowMatchingMoE

Main module implementing the FM-MoE layer.

#### Methods

**`forward(x, return_aux_loss=True)`**
- **Input**: `x` - Tensor of shape `(batch_size, seq_len, input_dim)`
- **Output**: Tuple of `(output, aux_loss)` where output has same shape as input

**`get_expert_usage_stats(x)`**
- **Input**: `x` - Tensor of shape `(batch_size, seq_len, input_dim)`
- **Output**: Dictionary containing `expert_probs`, `expert_selections`, `balance_score`

### FlowMatchingExpert

Individual flow matching expert network.

#### Methods

**`forward(x, t)`**
- Computes velocity field at position `x` and time `t`

**`flow_transform(x, steps=10)`**
- Applies complete flow transformation via Euler integration

---

## Configuration

### Recommended Configurations

#### Small Model (Research/Prototyping)
```python
config = FlowMatchingMoEConfig(
    input_dim=256,
    hidden_dim=512,
    num_experts=4,
    num_selected=1,
    flow_steps=5,
)
```

#### Base Model
```python
config = FlowMatchingMoEConfig(
    input_dim=512,
    hidden_dim=2048,
    num_experts=8,
    num_selected=2,
    flow_steps=10,
)
```

#### Large Model
```python
config = FlowMatchingMoEConfig(
    input_dim=1024,
    hidden_dim=4096,
    num_experts=16,
    num_selected=2,
    flow_steps=15,
)
```

### Hyperparameter Guidelines

| Parameter | Low | Medium | High | Notes |
|-----------|-----|--------|------|-------|
| `flow_steps` | 5 | 10 | 20 | More steps = higher accuracy, more compute |
| `num_experts` | 4 | 8 | 16 | Scale with model capacity |
| `num_selected` | 1 | 2 | 4 | Trade-off: efficiency vs. expressiveness |
| `load_balance_loss_coef` | 0.001 | 0.01 | 0.1 | Increase if experts collapse |

---

## Experimental Results

### Computational Complexity

| Component | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Router | O(B * L * E) | O(E * d) |
| Per Expert | O(B * L * N * d * h) | O(d * h) |
| Total | O(B * L * k * N * d * h) | O(E * d * h) |

Where: B = batch size, L = sequence length, E = num experts, k = num selected, N = flow steps, d = input dim, h = hidden dim

### Memory Efficiency

Compared to dense models with equivalent expressiveness, FM-MoE achieves:
- **Active Parameters**: k/E fraction of total parameters per forward pass
- **Memory Scaling**: Sub-linear with number of experts due to sparse activation

---

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{gomez2024fmmoe,
  author       = {Gomez, Kye},
  title        = {Flow Matching Mixture of Experts: Continuous Normalizing Flows for Sparse Expert Networks},
  year         = {2024},
  publisher    = {The Swarm Corporation},
  url          = {https://github.com/The-Swarm-Corporation/MoF}
}
```

### Related Works

```bibtex
@article{lipman2023flow,
  title={Flow Matching for Generative Modeling},
  author={Lipman, Yaron and Chen, Ricky TQ and Ben-Hamu, Heli and Nickel, Maximilian and Le, Matt},
  journal={International Conference on Learning Representations},
  year={2023}
}

@article{fedus2022switch,
  title={Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity},
  author={Fedus, William and Zoph, Barret and Shazeer, Noam},
  journal={Journal of Machine Learning Research},
  volume={23},
  number={120},
  pages={1--39},
  year={2022}
}

@article{shazeer2017outrageously,
  title={Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer},
  author={Shazeer, Noam and Mirhoseini, Azalia and Maziarz, Krzysztof and Davis, Andy and Le, Quoc and Hinton, Geoffrey and Dean, Jeff},
  journal={International Conference on Learning Representations},
  year={2017}
}

@article{chen2018neural,
  title={Neural Ordinary Differential Equations},
  author={Chen, Ricky TQ and Rubanova, Yulia and Bettencourt, Jesse and Duvenaud, David K},
  journal={Advances in Neural Information Processing Systems},
  volume={31},
  year={2018}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

This implementation builds upon foundational work in mixture of experts architectures and continuous normalizing flows. We acknowledge the contributions of the broader research community in advancing these methodologies.

---

## Author

**Kye Gomez**  
The Swarm Corporation  
[GitHub](https://github.com/The-Swarm-Corporation)
