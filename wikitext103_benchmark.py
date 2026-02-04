"""
WikiText-103 Perplexity Benchmark for Flow Matching MoE

Compares three architectures:
1. Dense Transformer (MLP FFN)
2. Standard MoE (MLP experts with top-k routing)
3. FM-MoE (Flow Matching experts with top-k routing)

Usage:
    python benchmarks/wikitext103_benchmark.py --model fm_moe --epochs 10
    python benchmarks/wikitext103_benchmark.py --model all --epochs 5 --quick

Author: MoF Research
Date: 2026-02-04
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mof.main import FlowMatchingMoE, FlowMatchingMoEConfig


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TransformerConfig:
    """Configuration for the benchmark transformer."""

    vocab_size: int = 50257  # GPT-2 vocab size
    max_seq_len: int = 256
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512  # FFN hidden dimension
    dropout: float = 0.1

    # MoE specific
    num_experts: int = 8
    num_selected: int = 2
    flow_steps: int = 10

    # Training
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: Optional[int] = None

    # Device
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )


# =============================================================================
# Standard MoE Baseline (MLP Experts)
# =============================================================================


class MLPExpert(nn.Module):
    """Standard MLP expert for baseline MoE."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StandardMoE(nn.Module):
    """
    Standard Mixture of Experts with MLP experts.
    Used as baseline comparison for FM-MoE.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        num_selected: int = 2,
        dropout: float = 0.1,
        load_balance_coef: float = 0.01,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_selected = num_selected
        self.load_balance_coef = load_balance_coef

        # Router
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        nn.init.normal_(self.gate.weight, std=0.01)

        # MLP experts
        self.experts = nn.ModuleList(
            [MLPExpert(d_model, d_ff, dropout) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, d_model = x.shape

        # Compute routing
        logits = self.gate(x)  # (B, S, E)
        probs = F.softmax(logits, dim=-1)

        # Top-k selection
        top_k_probs, top_k_indices = torch.topk(probs, self.num_selected, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # Process through experts
        x_flat = x.view(-1, d_model)
        top_k_probs_flat = top_k_probs.view(-1, self.num_selected)
        top_k_indices_flat = top_k_indices.view(-1, self.num_selected)
        output_flat = torch.zeros_like(x_flat)

        for expert_id in range(self.num_experts):
            expert_mask = top_k_indices_flat == expert_id
            if not expert_mask.any():
                continue

            positions, selection_indices = torch.where(expert_mask)
            if len(positions) == 0:
                continue

            expert_inputs = x_flat[positions]
            expert_weights = top_k_probs_flat[positions, selection_indices]
            expert_outputs = self.experts[expert_id](expert_inputs)
            weighted_outputs = expert_outputs * expert_weights.unsqueeze(-1)
            output_flat.index_add_(0, positions, weighted_outputs)

        output = output_flat.view(batch_size, seq_len, d_model)

        # Load balancing loss
        aux_loss = None
        if self.training:
            mean_probs = probs.mean(dim=[0, 1])
            target = 1.0 / self.num_experts
            aux_loss = (
                self.load_balance_coef
                * self.num_experts
                * torch.sum((mean_probs - target) ** 2)
            )

        return output, aux_loss

    def get_expert_usage_stats(self, x: torch.Tensor) -> dict:
        """Get expert usage statistics."""
        with torch.no_grad():
            logits = self.gate(x)
            probs = F.softmax(logits, dim=-1)
            _, top_k_indices = torch.topk(probs, self.num_selected, dim=-1)

            selections = torch.zeros(self.num_experts, device=x.device)
            for expert_id in range(self.num_experts):
                selections[expert_id] = (top_k_indices == expert_id).sum().float()

            mean_probs = probs.mean(dim=[0, 1])
            ideal = selections.sum() / self.num_experts
            balance_score = 1.0 - (selections - ideal).abs().sum() / (
                2 * selections.sum()
            )

            return {
                "expert_probs": mean_probs.cpu().numpy(),
                "expert_selections": selections.cpu().numpy(),
                "balance_score": balance_score.item(),
            }


class DenseFFN(nn.Module):
    """Standard dense FFN layer."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return self.net(x), None


# =============================================================================
# Transformer Components
# =============================================================================


class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, S, D = x.shape

        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, S, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scale = self.head_dim**-0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, S, D)
        out = self.out(out)

        return out


class TransformerBlock(nn.Module):
    """Single transformer block with configurable FFN type."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_module: nn.Module,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = ffn_module
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Self-attention with residual
        x = x + self.dropout(self.attn(self.ln1(x), mask))

        # FFN with residual
        ffn_out, aux_loss = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_out)

        return x, aux_loss


class BenchmarkTransformer(nn.Module):
    """
    Transformer for language modeling benchmark.
    Supports dense FFN, standard MoE, and FM-MoE.
    """

    def __init__(self, config: TransformerConfig, ffn_type: str = "dense"):
        super().__init__()
        self.config = config
        self.ffn_type = ffn_type

        # Token and position embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # Create FFN modules based on type
        self.blocks = nn.ModuleList()
        for _ in range(config.n_layers):
            if ffn_type == "dense":
                ffn = DenseFFN(config.d_model, config.d_ff, config.dropout)
            elif ffn_type == "standard_moe":
                ffn = StandardMoE(
                    d_model=config.d_model,
                    d_ff=config.d_ff,
                    num_experts=config.num_experts,
                    num_selected=config.num_selected,
                    dropout=config.dropout,
                )
            elif ffn_type == "fm_moe":
                fm_config = FlowMatchingMoEConfig(
                    input_dim=config.d_model,
                    hidden_dim=config.d_ff,
                    num_experts=config.num_experts,
                    num_selected=config.num_selected,
                    flow_steps=config.flow_steps,
                    dropout=config.dropout,
                )
                ffn = FlowMatchingMoE(fm_config)
            else:
                raise ValueError(f"Unknown FFN type: {ffn_type}")

            block = TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                ffn_module=ffn,
                dropout=config.dropout,
            )
            self.blocks.append(block)

        # Output head
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie embeddings
        self.head.weight = self.token_emb.weight

        # Initialize
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        B, S = input_ids.shape
        device = input_ids.device

        # Embeddings
        pos = torch.arange(S, device=device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.dropout(x)

        # Causal mask
        mask = torch.tril(torch.ones(S, S, device=device)).unsqueeze(0).unsqueeze(0)

        # Transformer blocks
        total_aux_loss = 0.0
        aux_count = 0
        for block in self.blocks:
            x, aux_loss = block(x, mask)
            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss
                aux_count += 1

        # Output
        x = self.ln_f(x)
        logits = self.head(x)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )

        aux_loss_mean = total_aux_loss / max(aux_count, 1) if aux_count > 0 else None

        return logits, loss, aux_loss_mean

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Count parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_emb.weight.numel()
            n_params -= self.pos_emb.weight.numel()
        return n_params

    def get_expert_stats(self, x: torch.Tensor) -> List[dict]:
        """Get expert usage stats from all MoE layers."""
        if self.ffn_type == "dense":
            return []

        stats = []
        for i, block in enumerate(self.blocks):
            if hasattr(block.ffn, "get_expert_usage_stats"):
                # Need to pass through embeddings first
                stat = block.ffn.get_expert_usage_stats(x)
                stat["layer"] = i
                stats.append(stat)
        return stats


# =============================================================================
# Dataset
# =============================================================================


class WikiText103Dataset(Dataset):
    """WikiText-103 dataset for language modeling."""

    def __init__(
        self,
        split: str = "train",
        max_seq_len: int = 256,
        tokenizer=None,
    ):
        self.max_seq_len = max_seq_len

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")

        # Load dataset
        print(f"Loading WikiText-103 {split} split...")
        dataset = load_dataset("wikitext", "wikitext-103-v1", split=split)

        # Load tokenizer
        if tokenizer is None:
            try:
                from transformers import GPT2Tokenizer

                tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                tokenizer.pad_token = tokenizer.eos_token
            except ImportError:
                raise ImportError(
                    "Please install transformers: pip install transformers"
                )

        self.tokenizer = tokenizer

        # Tokenize all text
        print("Tokenizing...")
        all_text = "\n".join([t for t in dataset["text"] if t.strip()])
        self.tokens = tokenizer.encode(all_text)

        # Calculate number of sequences
        self.n_sequences = (len(self.tokens) - 1) // max_seq_len
        print(f"Created {self.n_sequences} sequences of length {max_seq_len}")

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.max_seq_len
        end = start + self.max_seq_len + 1

        chunk = self.tokens[start:end]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)

        return x, y


def create_dataloaders(
    config: TransformerConfig,
    tokenizer=None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders."""

    train_ds = WikiText103Dataset("train", config.max_seq_len, tokenizer)
    val_ds = WikiText103Dataset("validation", config.max_seq_len, train_ds.tokenizer)
    test_ds = WikiText103Dataset("test", config.max_seq_len, train_ds.tokenizer)

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader


# =============================================================================
# Training
# =============================================================================


def get_lr(step: int, warmup_steps: int, max_lr: float, total_steps: int) -> float:
    """Learning rate schedule with warmup and cosine decay."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    decay_steps = total_steps - warmup_steps
    decay_step = step - warmup_steps
    return max_lr * 0.5 * (1 + math.cos(math.pi * decay_step / decay_steps))


@torch.no_grad()
def evaluate(
    model: BenchmarkTransformer,
    dataloader: DataLoader,
    device: str,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate model on a dataset."""
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    total_aux_loss = 0.0
    aux_count = 0

    for i, (x, y) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break

        x, y = x.to(device), y.to(device)
        _, loss, aux_loss = model(x, y)

        n_tokens = (y != -100).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

        if aux_loss is not None:
            total_aux_loss += aux_loss.item()
            aux_count += 1

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    results = {
        "loss": avg_loss,
        "perplexity": perplexity,
    }

    if aux_count > 0:
        results["aux_loss"] = total_aux_loss / aux_count

    return results


def train_epoch(
    model: BenchmarkTransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    config: TransformerConfig,
    global_step: int,
    total_steps: int,
) -> Tuple[Dict[str, float], int]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_aux_loss = 0.0
    total_tokens = 0
    aux_count = 0
    log_interval = 100

    start_time = time.time()

    for batch_idx, (x, y) in enumerate(dataloader):
        if config.max_steps is not None and global_step >= config.max_steps:
            break

        # Update learning rate
        lr = get_lr(global_step, config.warmup_steps, config.learning_rate, total_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = x.to(device), y.to(device)

        # Forward pass
        _, loss, aux_loss = model(x, y)

        # Add auxiliary loss if present
        total_loss_step = loss
        if aux_loss is not None:
            total_loss_step = loss + aux_loss
            total_aux_loss += aux_loss.item()
            aux_count += 1

        # Backward pass
        optimizer.zero_grad()
        total_loss_step.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Track metrics
        n_tokens = (y != -100).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
        global_step += 1

        # Log progress
        if (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = total_tokens / elapsed
            avg_loss = total_loss / total_tokens
            ppl = math.exp(avg_loss)

            aux_str = ""
            if aux_count > 0:
                aux_str = f", aux_loss: {total_aux_loss/aux_count:.4f}"

            print(
                f"  Epoch {epoch} | Batch {batch_idx+1}/{len(dataloader)} | "
                f"Loss: {avg_loss:.4f} | PPL: {ppl:.2f}{aux_str} | "
                f"LR: {lr:.2e} | {tokens_per_sec:.0f} tok/s"
            )

    avg_loss = total_loss / total_tokens
    results = {
        "loss": avg_loss,
        "perplexity": math.exp(avg_loss),
        "tokens_per_sec": total_tokens / (time.time() - start_time),
    }
    if aux_count > 0:
        results["aux_loss"] = total_aux_loss / aux_count

    return results, global_step


def train_model(
    model: BenchmarkTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TransformerConfig,
    epochs: int,
    model_name: str,
) -> Dict:
    """Train a model and return results."""
    device = config.device
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Calculate total steps
    if config.max_steps is not None:
        total_steps = config.max_steps
    else:
        total_steps = len(train_loader) * epochs

    print(f"\n{'='*70}")
    print(f"Training {model_name}")
    print(f"{'='*70}")
    print(f"Parameters: {model.get_num_params():,}")
    print(f"Device: {device}")
    print(f"Total steps: {total_steps}")
    print()

    # Training loop
    global_step = 0
    best_val_ppl = float("inf")
    history = {
        "train_loss": [],
        "train_ppl": [],
        "val_loss": [],
        "val_ppl": [],
        "expert_stats": [],
    }

    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")

        # Train
        train_results, global_step = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            config,
            global_step,
            total_steps,
        )
        history["train_loss"].append(train_results["loss"])
        history["train_ppl"].append(train_results["perplexity"])

        # Validate
        val_results = evaluate(model, val_loader, device, max_batches=50)
        history["val_loss"].append(val_results["loss"])
        history["val_ppl"].append(val_results["perplexity"])

        # Get expert stats if applicable
        if model.ffn_type != "dense":
            sample_x, _ = next(iter(val_loader))
            sample_x = sample_x.to(device)
            with torch.no_grad():
                # Get hidden states after embedding
                pos = torch.arange(sample_x.shape[1], device=device).unsqueeze(0)
                hidden = model.token_emb(sample_x) + model.pos_emb(pos)
                stats = model.get_expert_stats(hidden)
                if stats:
                    history["expert_stats"].append(
                        {
                            "epoch": epoch,
                            "balance_scores": [s["balance_score"] for s in stats],
                        }
                    )

        # Track best
        if val_results["perplexity"] < best_val_ppl:
            best_val_ppl = val_results["perplexity"]

        print(
            f"\n  Train Loss: {train_results['loss']:.4f} | Train PPL: {train_results['perplexity']:.2f}"
        )
        print(
            f"  Val Loss: {val_results['loss']:.4f} | Val PPL: {val_results['perplexity']:.2f}"
        )
        print(f"  Best Val PPL: {best_val_ppl:.2f}")

        if config.max_steps is not None and global_step >= config.max_steps:
            print(f"Reached max_steps ({config.max_steps}), stopping.")
            break

    return {
        "model_name": model_name,
        "ffn_type": model.ffn_type,
        "num_params": model.get_num_params(),
        "best_val_ppl": best_val_ppl,
        "final_train_ppl": history["train_ppl"][-1],
        "final_val_ppl": history["val_ppl"][-1],
        "history": history,
    }


# =============================================================================
# Main Benchmark
# =============================================================================


def run_benchmark(
    models_to_run: List[str],
    epochs: int,
    config: TransformerConfig,
    output_dir: str = "evals",
) -> Dict:
    """Run the full benchmark suite."""

    print("=" * 70)
    print("WikiText-103 Perplexity Benchmark")
    print("=" * 70)
    print(f"Models to benchmark: {models_to_run}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Sequence length: {config.max_seq_len}")
    print(f"Model dim: {config.d_model}")
    print(f"Layers: {config.n_layers}")
    print(f"Experts: {config.num_experts} (select {config.num_selected})")
    print(f"Flow steps: {config.flow_steps}")
    print()

    # Create dataloaders
    print("Loading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(config)
    print()

    # Run each model
    results = {}

    for model_type in models_to_run:
        print(f"\n{'#'*70}")
        print(f"# Benchmarking: {model_type.upper()}")
        print(f"{'#'*70}")

        model = BenchmarkTransformer(config, ffn_type=model_type)

        result = train_model(
            model, train_loader, val_loader, config, epochs, model_type
        )

        # Final test evaluation
        print(f"\nFinal test evaluation for {model_type}...")
        test_results = evaluate(model, test_loader, config.device)
        result["test_loss"] = test_results["loss"]
        result["test_ppl"] = test_results["perplexity"]
        print(f"  Test PPL: {test_results['perplexity']:.2f}")

        results[model_type] = result

        # Clear memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(
        f"{'Model':<15} {'Params':<12} {'Train PPL':<12} {'Val PPL':<12} {'Test PPL':<12}"
    )
    print("-" * 70)

    for name, res in results.items():
        print(
            f"{name:<15} {res['num_params']:<12,} {res['final_train_ppl']:<12.2f} "
            f"{res['final_val_ppl']:<12.2f} {res['test_ppl']:<12.2f}"
        )

    print("=" * 70)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Convert numpy arrays for JSON serialization
    def convert_for_json(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        return obj

    results_file = os.path.join(output_dir, f"wikitext103_benchmark_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(convert_for_json(results), f, indent=2)
    print(f"\nResults saved to: {results_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="WikiText-103 Perplexity Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["dense", "standard_moe", "fm_moe", "all"],
        help="Model type to benchmark",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=256, help="Sequence length")
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument(
        "--n_layers", type=int, default=4, help="Number of transformer layers"
    )
    parser.add_argument(
        "--num_experts", type=int, default=8, help="Number of experts for MoE models"
    )
    parser.add_argument(
        "--num_selected",
        type=int,
        default=2,
        help="Number of experts to select per token",
    )
    parser.add_argument(
        "--flow_steps",
        type=int,
        default=10,
        help="Number of flow integration steps for FM-MoE",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Max training steps (overrides epochs)",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick run with fewer steps for testing"
    )
    parser.add_argument(
        "--output_dir", type=str, default="evals", help="Output directory for results"
    )

    args = parser.parse_args()

    # Build config
    config = TransformerConfig(
        max_seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=4,
        n_layers=args.n_layers,
        d_ff=args.d_model * 2,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_experts=args.num_experts,
        num_selected=args.num_selected,
        flow_steps=args.flow_steps,
        max_steps=args.max_steps,
    )

    # Quick mode for testing
    if args.quick:
        config.max_steps = 500
        args.epochs = 1
        print("Quick mode: Running 500 steps only")

    # Determine models to run
    if args.model == "all":
        models = ["dense", "standard_moe", "fm_moe"]
    else:
        models = [args.model]

    # Run benchmark
    run_benchmark(models, args.epochs, config, args.output_dir)


if __name__ == "__main__":
    main()
