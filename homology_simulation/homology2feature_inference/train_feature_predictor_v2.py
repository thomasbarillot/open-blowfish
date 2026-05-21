#!/usr/bin/env python3
"""
Encoder-only feature predictor (v3, classifier-only).

Input entropy map: (T, 20) float32. Up to 50 n_neighbour rows, 20 h0_birth bins.

Output: per-(neighbour, feature) classification over {-1, 0, +1}.
Loss: class-weighted cross-entropy with the +1 class up-weighted.
"""

import argparse
import sys
import threading
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


S3_BUCKET = "homology-experiment"
S3_PREFIX = "simulation_dataset_v2"
S3_MODEL_KEY = "model/model_v3.pt"
S3_TB_PREFIX = "model/runs_v3"
S3_PLOTS_PREFIX = "model/plots_v3"

# Data geometry
MAX_NEIGHBORS = 50
MAX_FEATURES = 64
N_H0_BINS = 20
N_CLASSES = 3  # {-1, 0, +1} -> {0, 1, 2}

# Default model geometry (overridable via CLI)
DEFAULT_D_MODEL = 384
DEFAULT_ENC_LAYERS = 4
DEFAULT_HEADS = 6
DEFAULT_D_FF = 1536
DEFAULT_DROPOUT = 0.1


def parse_args():
    parser = argparse.ArgumentParser(description="Train encoder-only classifier feature predictor (v3)")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).parent / "outputs" / "simulation_dataset_v2"),
        help="Local dir or s3:// URI with entropy_maps.npy and label_matrices.npy",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).parent / "outputs" / "model_v3"),
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-3,
                        help="Loss changes within +/- this band don't move the patience counter.")
    parser.add_argument("--prefix-truncation", action="store_true",
                        help="Random prefix truncation during training")
    parser.add_argument("--min-prefix", type=int, default=3)
    parser.add_argument("--min-t-star", type=int, default=2,
                        help="Drop dataset samples whose informative-prefix length T* < this.")
    parser.add_argument("--class-weight-strategy", type=str, default="auto",
                        choices=["auto", "uniform"],
                        help='"auto" computes inverse-frequency weights from the train cells; '
                             '"uniform" uses [1, 1, 1].')
    parser.add_argument("--d-model", type=int, default=DEFAULT_D_MODEL)
    parser.add_argument("--enc-layers", type=int, default=DEFAULT_ENC_LAYERS)
    parser.add_argument("--heads", type=int, default=DEFAULT_HEADS)
    parser.add_argument("--d-ff", type=int, default=DEFAULT_D_FF)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--predict", type=str, default=None,
                        help="Path to dump_df0.pkl for inference mode (requires trained model)")
    parser.add_argument("--upload-model-s3", action="store_true")
    parser.add_argument("--model-from-s3", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision on CUDA")
    parser.add_argument("--amp-dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--tb-upload-every", type=int, default=10)
    return parser.parse_args()


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _find_large_volume_dir() -> Path:
    import shutil
    candidates = [Path("/av"), Path("/data"), Path("/mnt/data"), Path("/mnt")]
    best_path = None
    best_free = 0
    for p in candidates:
        if p.exists():
            free = shutil.disk_usage(p).free
            if free > best_free:
                best_free = free
                best_path = p
    if best_path and best_free > 10 * 1024**3:
        cache = best_path / "entropy_dataset_cache_v2"
        cache.mkdir(parents=True, exist_ok=True)
        return cache
    import tempfile
    return Path(tempfile.mkdtemp(prefix="entropy_dataset_v2_"))


def _get_s3_client():
    import boto3

    from config import AWS_PROFILE_NAME
    return boto3.Session(profile_name=AWS_PROFILE_NAME).client("s3")


def resolve_data_dir(data_dir: str) -> Path:
    if not data_dir.startswith("s3://"):
        return Path(data_dir)
    parts = data_dir.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    s3 = _get_s3_client()
    local_dir = _find_large_volume_dir()
    for fname in ("entropy_maps.npy", "label_matrices.npy", "metadata.csv"):
        s3_key = f"{prefix}/{fname}" if prefix else fname
        local_path = local_dir / fname
        if local_path.exists():
            print(f"Using cached {local_path}")
            continue
        print(f"Downloading s3://{bucket}/{s3_key} -> {local_path}")
        s3.download_file(bucket, s3_key, str(local_path))
    print(f"Dataset cached at {local_dir}")
    return local_dir


def upload_model_to_s3(local_path: Path, bucket: str = S3_BUCKET, key: str = S3_MODEL_KEY):
    s3 = _get_s3_client()
    print(f"Uploading model to s3://{bucket}/{key}")
    s3.upload_file(str(local_path), bucket, key)


def download_model_from_s3(local_path: Path, bucket: str = S3_BUCKET, key: str = S3_MODEL_KEY):
    s3 = _get_s3_client()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading model from s3://{bucket}/{key}")
    s3.download_file(bucket, key, str(local_path))


def upload_dir_to_s3(local_dir: Path, bucket: str, prefix: str):
    s3 = _get_s3_client()
    count = 0
    for path in sorted(local_dir.rglob("*")):
        if path.is_file():
            key = f"{prefix}/{path.relative_to(local_dir)}"
            s3.upload_file(str(path), bucket, key)
            count += 1
    print(f"Uploaded {count} files to s3://{bucket}/{prefix}/")


def sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * -(np.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class RowEncoder(nn.Module):
    """Per-row encoder: (B, T, 20) -> (B, T, d_model). Shared across rows."""

    def __init__(self, d_model: int, n_h0_bins: int = N_H0_BINS):
        super().__init__()
        self.conv = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.mlp = nn.Sequential(
            nn.Linear(32 * n_h0_bins, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x):
        b, t, n_h0 = x.shape
        x = x.reshape(b * t, 1, n_h0)
        x = self.conv(x)
        x = F.gelu(x)
        x = x.reshape(b, t, -1)
        return self.mlp(x)


class RowTransformerEncoder(nn.Module):
    """Pre-LN, GELU, batch_first transformer encoder with bidirectional attention along T."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int, d_ff: int, dropout: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.stack = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x, key_padding_mask=None):
        return self.stack(x, src_key_padding_mask=key_padding_mask)


class FeaturePredictorV2(nn.Module):
    """
    Encoder-only per-cell classifier.

    forward(entropy_map, valid_mask=None) -> (B, T, 64, 3) class logits.

    valid_mask is forwarded to the encoder's key_padding_mask so attention
    ignores padded rows. The forward output is unmasked; the loss restricts
    supervision to valid rows.
    """

    def __init__(
        self,
        d_model: int = DEFAULT_D_MODEL,
        enc_layers: int = DEFAULT_ENC_LAYERS,
        heads: int = DEFAULT_HEADS,
        d_ff: int = DEFAULT_D_FF,
        dropout: float = DEFAULT_DROPOUT,
        seq_len: int = MAX_NEIGHBORS,
        n_features: int = MAX_FEATURES,
        n_h0_bins: int = N_H0_BINS,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.row_enc = RowEncoder(d_model=d_model, n_h0_bins=n_h0_bins)
        self.register_buffer(
            "pos_enc",
            sinusoidal_positional_encoding(seq_len, d_model),
            persistent=False,
        )
        self.row_attn = RowTransformerEncoder(
            d_model=d_model, n_heads=heads, n_layers=enc_layers, d_ff=d_ff, dropout=dropout,
        )
        # Output-side neighbour position embedding. Concat'd with row_context
        # before the head so the classifier sees "I am at neighbour t" directly.
        self.head_pos_emb = nn.Parameter(torch.randn(seq_len, d_model) * 0.02)
        self.class_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_features * N_CLASSES),
        )

    def forward(self, entropy_map, valid_mask=None):
        t = entropy_map.size(1)
        x = self.row_enc(entropy_map)
        x = x + self.pos_enc[:t].unsqueeze(0)

        if valid_mask is not None:
            kpm = ~valid_mask
            empty = ~valid_mask.any(dim=1)
            if empty.any():
                kpm = kpm.clone()
                kpm[empty, 0] = False
        else:
            kpm = None

        h = self.row_attn(x, key_padding_mask=kpm)                          # (B, T, d_model)
        pos = self.head_pos_emb[:t].unsqueeze(0).expand(h.size(0), -1, -1)  # (B, T, d_model)
        h_pos = torch.cat([h, pos], dim=-1)                                 # (B, T, 2*d_model)
        b = h_pos.size(0)
        return self.class_head(h_pos).view(b, t, self.n_features, N_CLASSES)


def compute_inverse_frequency_weights(
    dataset, indices, sample_cap: int = 20000, max_weight: float = 8.0,
) -> torch.Tensor:
    """
    Inverse-frequency class weights over the informative region of train cells.

    Counts each class on the truncated supervised prefix (rows < T*) for the
    sampled training indices, then weights = 1 / freq. Normalised to mean=1
    across the 3 classes so the loss magnitude stays comparable when classes
    are switched. Clamped to `max_weight` to prevent rare-class blow-up.

    The dataset must expose `signed_targets` and `t_star` tensors.
    """
    sample = (
        indices if len(indices) <= sample_cap
        else np.random.default_rng(0).choice(indices, size=sample_cap, replace=False)
    )
    counts = np.zeros(N_CLASSES, dtype=np.float64)
    for i in sample:
        sgn = dataset.signed_targets[i].numpy()
        t_star = int(dataset.t_star[i])
        block = sgn[:t_star]                                                  # (t_star, F)
        for c, val in enumerate((-1.0, 0.0, 1.0)):
            counts[c] += (block == val).sum()
    total = counts.sum()
    if (counts == 0).any() or total == 0:
        return torch.tensor([1.0, 1.0, 1.0])
    freqs = counts / total
    inv = 1.0 / freqs
    inv = np.clip(inv, 0.0, max_weight)
    inv = inv * (N_CLASSES / inv.sum())   # mean = 1
    print(f"[class_weights] counts {counts.tolist()} | freqs {freqs.tolist()} | weights {inv.tolist()}")
    return torch.tensor(inv, dtype=torch.float32)


def class_loss_and_acc(class_logits, target_signed, mask, class_weights: torch.Tensor):
    """
    class_logits:  (B, T, 64, 3) float
    target_signed: (B, T, 64) float in {-1, 0, +1}
    mask:          (B, T) bool — True on rows that contribute to the loss
    class_weights: (3,) float — per-class CE weights for [-1, 0, +1].

    Class-weighted cross-entropy. Denominator uses the same weights as the
    numerator (proper weighted average): doubling a weight redistributes
    gradient share but doesn't trivially rescale the loss magnitude.

    Returns dict with:
      - loss:       weighted CE
      - cell_acc:   per-cell exact-match rate using argmax(class_logits)
      - pos_recall: fraction of true +1 cells correctly predicted as +1
      - neg_recall: fraction of true -1 cells correctly predicted as -1
    """
    target_idx = (target_signed + 1).long()                                  # (B, T, F) in {0, 1, 2}
    log_probs = F.log_softmax(class_logits, dim=-1)                          # (B, T, F, 3)
    nll = -log_probs.gather(-1, target_idx.unsqueeze(-1)).squeeze(-1)        # (B, T, F)

    cw = class_weights.to(device=class_logits.device, dtype=class_logits.dtype)
    mask_f = mask.to(class_logits.dtype)
    m_cell = mask_f.unsqueeze(-1)                                            # (B, T, 1)
    sample_weight = cw[target_idx] * m_cell                                  # (B, T, F)
    ce_denom = sample_weight.sum().clamp(min=1.0)
    loss = (nll * sample_weight).sum() / ce_denom

    with torch.no_grad():
        pred_class = class_logits.argmax(dim=-1) - 1                         # (B, T, F) in {-1, 0, +1}
        correct = (pred_class == target_signed.long()).to(class_logits.dtype)
        cell_count_denom = (m_cell.sum() * class_logits.size(-2)).clamp(min=1.0)
        acc = (correct * m_cell).sum() / cell_count_denom

        is_pos = (target_signed == 1) & m_cell.bool()
        pos_total = is_pos.sum().clamp(min=1)
        pos_correct = ((pred_class == 1) & is_pos).sum()
        pos_recall = pos_correct.float() / pos_total.float()

        is_neg = (target_signed == -1) & m_cell.bool()
        neg_total = is_neg.sum().clamp(min=1)
        neg_correct = ((pred_class == -1) & is_neg).sum()
        neg_recall = neg_correct.float() / neg_total.float()

    return {
        "loss": loss,
        "cell_acc": acc,
        "pos_recall": pos_recall,
        "neg_recall": neg_recall,
    }


def _move_batch(batch, device):
    maps, signed_targets = batch
    return (
        maps.to(device, non_blocking=True),
        signed_targets.to(device, non_blocking=True),
    )


def _informative_lengths(signed_targets: torch.Tensor) -> torch.Tensor:
    """
    T* per sample: smallest prefix such that rows [T*, T) all equal row T*-1.
    signed_targets: (B, T, F). Returns (B,) long in [1, T].
    """
    b, t, _ = signed_targets.shape
    if t < 2:
        return torch.full((b,), t, dtype=torch.long, device=signed_targets.device)
    row_equal = (signed_targets[:, 1:] == signed_targets[:, :-1]).all(dim=-1)
    trailing = row_equal.flip(-1).int().cumprod(dim=-1).sum(dim=-1)
    return (t - trailing).clamp(min=1, max=t)


def _prepare_batch(maps, signed_targets, random_truncate: bool, min_prefix: int):
    """
    Compute T_eff, zero out map / target rows beyond it, return mask.

    The returned `mask` is True only on rows that are (a) within T_eff, and
    (b) carry non-zero input signal. Empty input rows (rows 0 and 1 are zero
    by construction in the simulation dataset) are excluded from supervision.
    """
    b, t, _ = maps.shape
    t_star = _informative_lengths(signed_targets)
    if random_truncate:
        hi = t_star
        lo = torch.minimum(torch.full_like(t_star, min_prefix), hi)
        u = torch.rand(b, device=maps.device)
        t_eff = (lo + (u * (hi - lo + 1).float()).long()).clamp(min=1, max=t)
    else:
        t_eff = t_star

    ar = torch.arange(t, device=maps.device).unsqueeze(0)
    informative = ar < t_eff.unsqueeze(1)
    has_input = (maps.abs().sum(dim=-1) > 0)
    mask = informative & has_input

    maps = maps * informative.unsqueeze(-1).to(maps.dtype)
    signed_targets = signed_targets * informative.unsqueeze(-1).to(signed_targets.dtype)
    return maps, signed_targets, mask, t_eff


def train_one_epoch(model, loader, optimizer, scheduler, device, args, scaler=None, use_amp=False):
    model.train()
    running = {"loss": 0.0, "cell_acc": 0.0, "pos_recall": 0.0, "neg_recall": 0.0, "t_eff": 0.0}
    n_samples = 0
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if use_amp else torch.autocast(device_type=device.type, enabled=False)
    )

    for batch in loader:
        maps, signed_targets = _move_batch(batch, device)
        maps, signed_targets, mask, t_eff = _prepare_batch(
            maps, signed_targets,
            random_truncate=args.prefix_truncation, min_prefix=args.min_prefix,
        )
        optimizer.zero_grad()
        with amp_ctx:
            class_logits = model(maps, valid_mask=mask)
            metrics = class_loss_and_acc(
                class_logits, signed_targets, mask,
                class_weights=args.class_weights,
            )
        loss = metrics["loss"]
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer_stepped = True
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_stepped = scaler.get_scale() >= scale_before
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        if optimizer_stepped:
            scheduler.step()

        bs = maps.size(0)
        running["loss"] += float(metrics["loss"]) * bs
        running["cell_acc"] += float(metrics["cell_acc"]) * bs
        running["pos_recall"] += float(metrics["pos_recall"]) * bs
        running["neg_recall"] += float(metrics["neg_recall"]) * bs
        running["t_eff"] += float(t_eff.float().mean()) * bs
        n_samples += bs

    return {k: v / n_samples for k, v in running.items()}


def evaluate(model, loader, device, args, use_amp=False):
    model.eval()
    running = {"loss": 0.0, "cell_acc": 0.0, "pos_recall": 0.0, "neg_recall": 0.0, "t_eff": 0.0}
    n_samples = 0
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if use_amp else torch.autocast(device_type=device.type, enabled=False)
    )
    with torch.no_grad():
        for batch in loader:
            maps, signed_targets = _move_batch(batch, device)
            maps, signed_targets, mask, t_eff = _prepare_batch(
                maps, signed_targets, random_truncate=False, min_prefix=args.min_prefix,
            )
            with amp_ctx:
                class_logits = model(maps, valid_mask=mask)
                metrics = class_loss_and_acc(
                    class_logits, signed_targets, mask,
                    class_weights=args.class_weights,
                )
            bs = maps.size(0)
            running["loss"] += float(metrics["loss"]) * bs
            running["cell_acc"] += float(metrics["cell_acc"]) * bs
            running["pos_recall"] += float(metrics["pos_recall"]) * bs
            running["neg_recall"] += float(metrics["neg_recall"]) * bs
            running["t_eff"] += float(t_eff.float().mean()) * bs
            n_samples += bs
    return {k: v / n_samples for k, v in running.items()}


def build_lr_scheduler(optimizer, total_steps: int, warmup_steps: int, peak_lr: float, final_lr: float):
    import math

    def lr_lambda(step: int):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (final_lr + (peak_lr - final_lr) * cos) / peak_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def plot_training_history(history, output_dir: Path):
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    axes[0].plot(history["train_loss"], label="Train"); axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_title("Weighted CE loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(history["train_cell_acc"], label="Train"); axes[1].plot(history["val_cell_acc"], label="Val")
    axes[1].set_title("Cell accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    axes[2].plot(history["train_pos_recall"], label="Train"); axes[2].plot(history["val_pos_recall"], label="Val")
    axes[2].set_title("+1 recall"); axes[2].set_xlabel("Epoch"); axes[2].legend()
    axes[3].plot(history["train_neg_recall"], label="Train"); axes[3].plot(history["val_neg_recall"], label="Val")
    axes[3].set_title("-1 recall"); axes[3].set_xlabel("Epoch"); axes[3].legend()
    plt.tight_layout()
    fig.savefig(output_dir / "training_history.png", dpi=150)
    plt.close(fig)


def plot_reconstruction_comparison(model, dataset, device, output_dir: Path,
                                   n_examples: int = 8, indices=None):
    """
    Plot ground truth vs argmax reconstruction across the accuracy range.

    `indices`: optional list/array of dataset indices to sample from. Pass
    val indices to keep the plot honest (otherwise we'd be showing a mix of
    train and val samples and the per-sample accs in the panel titles
    wouldn't match the val metrics from training).
    """
    from torch.utils.data import DataLoader, Subset
    model.eval()
    if indices is None:
        pool = list(range(len(dataset)))
    else:
        pool = list(indices)
    n_eval = min(len(pool), 2000)
    rng = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(pool), generator=rng)[:n_eval].tolist()
    pick = [pool[i] for i in perm]
    sub = Subset(dataset, pick)
    loader = DataLoader(sub, batch_size=128, shuffle=False)

    all_recon, all_targets, all_maps, all_t_eff = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            maps, signed_targets = _move_batch(batch, device)
            maps, signed_targets, mask, t_eff = _prepare_batch(
                maps, signed_targets, random_truncate=False, min_prefix=1,
            )
            class_logits = model(maps, valid_mask=mask)                       # (B, T, F, 3)
            pred_class = (class_logits.argmax(dim=-1) - 1).to(torch.int8)     # (B, T, F) in {-1, 0, +1}
            all_recon.append(pred_class.cpu().numpy())
            all_targets.append(signed_targets.cpu().numpy().astype(np.int8))
            all_maps.append(maps.cpu().numpy())
            all_t_eff.append(t_eff.cpu().numpy())

    recon = np.concatenate(all_recon)            # (N, T, 64)
    targets = np.concatenate(all_targets)        # (N, T, 64)
    maps = np.concatenate(all_maps)              # (N, T, 20)
    t_eff_all = np.concatenate(all_t_eff)        # (N,)

    for i in range(len(recon)):
        te = int(t_eff_all[i])
        if te < recon.shape[1]:
            recon[i, te:, :] = 0

    per_sample_acc = np.zeros(len(recon), dtype=np.float32)
    for i in range(len(recon)):
        te = int(t_eff_all[i])
        per_sample_acc[i] = (recon[i, :te, :] == targets[i, :te, :]).mean()
    sorted_idx = np.argsort(per_sample_acc)
    picks = np.linspace(0, len(sorted_idx) - 1, n_examples, dtype=int)
    pick_idxs = sorted_idx[picks]

    fig, axes = plt.subplots(n_examples, 3, figsize=(18, 4 * n_examples))
    if n_examples == 1:
        axes = axes[np.newaxis, :]
    for row, i in enumerate(pick_idxs):
        te = int(t_eff_all[i])
        ax_gt = axes[row, 0]
        im = ax_gt.imshow(targets[i].T, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)
        ax_gt.axvline(te - 0.5, color="cyan", linestyle="--", linewidth=1.0)
        ax_gt.set_title(f"Ground truth (acc={per_sample_acc[i]:.3f}, T*={te})")
        ax_gt.set_ylabel("Feature ID"); ax_gt.set_xlabel("Neighbour")
        plt.colorbar(im, ax=ax_gt)

        ax_rc = axes[row, 1]
        im = ax_rc.imshow(recon[i].T, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)
        ax_rc.axvline(te - 0.5, color="cyan", linestyle="--", linewidth=1.0)
        ax_rc.set_title("Reconstructed (argmax)")
        ax_rc.set_ylabel("Feature ID"); ax_rc.set_xlabel("Neighbour")
        plt.colorbar(im, ax=ax_rc)

        ax_map = axes[row, 2]
        im = ax_map.imshow(maps[i], cmap="hot", aspect="auto")
        ax_map.axhline(te - 0.5, color="cyan", linestyle="--", linewidth=1.0)
        ax_map.set_title("Entropy map")
        ax_map.set_ylabel("n_neighbours"); ax_map.set_xlabel("h0 bin")
        plt.colorbar(im, ax=ax_map)

    plt.tight_layout()
    fig.savefig(output_dir / "reconstruction_comparison.png", dpi=150)
    plt.close(fig)


class MapLabelDataset(Dataset):
    """
    Yields (map, signed_target).

    map:           (n_rows, n_h0) float32, row-normalized.
    signed_target: (50, 64) float32, values in {-1, 0, +1}.
    """

    def __init__(self, maps_path: str, labels_path: str, min_t_star: int = 2):
        raw_maps = np.load(maps_path, mmap_mode="r")
        raw_labels = np.load(labels_path, mmap_mode="r")
        n = raw_maps.shape[0]
        assert raw_maps.ndim == 3, f"expected 3D entropy maps, got {raw_maps.shape}"
        assert raw_labels.shape == (n, MAX_NEIGHBORS, MAX_FEATURES), (
            f"label shape mismatch: {raw_labels.shape}"
        )

        n_rows, n_h0 = raw_maps.shape[1], raw_maps.shape[2]
        assert n_h0 == N_H0_BINS, f"h0 axis mismatch: expected {N_H0_BINS}, got {n_h0}"
        self.n_rows = n_rows

        chunk = 10_000
        maps_np = np.empty((n, n_rows, n_h0), dtype=np.float32)
        for i in range(0, n, chunk):
            block = np.asarray(raw_maps[i : i + chunk], dtype=np.float32)
            row_sums = block.sum(axis=-1, keepdims=True)
            row_sums = np.where(row_sums > 0, row_sums, 1.0)
            maps_np[i : i + chunk] = block / row_sums

        signed_np = np.empty((n, MAX_NEIGHBORS, MAX_FEATURES), dtype=np.float32)
        for i in range(0, n, chunk):
            signed_np[i : i + chunk] = np.asarray(raw_labels[i : i + chunk]).astype(np.float32)

        if MAX_NEIGHBORS >= 2:
            row_eq = (signed_np[:, 1:] == signed_np[:, :-1]).all(axis=-1)
            trailing = np.flip(row_eq, axis=-1).astype(np.int32).cumprod(axis=-1).sum(axis=-1)
            t_star_np = np.clip(MAX_NEIGHBORS - trailing, 1, MAX_NEIGHBORS).astype(np.int32)
        else:
            t_star_np = np.full((n,), MAX_NEIGHBORS, dtype=np.int32)

        if min_t_star > 1:
            keep_mask = t_star_np >= int(min_t_star)
            n_dropped = int((~keep_mask).sum())
            if n_dropped > 0:
                print(
                    f"[MapLabelDataset] dropping {n_dropped}/{n} samples with T* < {min_t_star} "
                    f"(degenerate inputs; no informative neighbour-row context)."
                )
                maps_np = maps_np[keep_mask]
                signed_np = signed_np[keep_mask]
                t_star_np = t_star_np[keep_mask]
                n = maps_np.shape[0]

        p_feat = (signed_np != 0).mean(axis=(1, 2)).astype(np.float32)

        self.maps = torch.from_numpy(maps_np)
        self.signed_targets = torch.from_numpy(signed_np)
        self.p_feat = torch.from_numpy(p_feat)
        self.t_star = torch.from_numpy(t_star_np.astype(np.int64))
        self.min_t_star = int(min_t_star)

    def __len__(self):
        return self.maps.size(0)

    def __getitem__(self, idx):
        return (self.maps[idx], self.signed_targets[idx])


def build_experimental_entropy_maps(data_path: str) -> tuple[list[np.ndarray], list[dict]]:
    """Build variable-length (T, 20) entropy maps from experimental OTPP data."""
    import pandas as pd
    from tqdm import tqdm

    print(f"Loading experimental data from {data_path}...")
    df = pd.read_pickle(data_path)
    print(f"Loaded {len(df)} rows")

    h0_bins = np.linspace(0.0, 1.5, N_H0_BINS + 1)
    max_nn = MAX_NEIGHBORS

    group_cols = ["tag_name", "doc_name", "pnum", "query_granularity", "corpus_granularity"]
    maps = []
    metadata = []

    for group_key, gdf in tqdm(df.groupby(group_cols), desc="Building experimental maps"):
        tag_name, doc_name, pnum, query_gran, corpus_gran = group_key
        entropy_map = np.zeros((max_nn, N_H0_BINS), dtype=np.float32)
        valid = gdf.dropna(subset=["nneighbours", "h0_dist"])
        if len(valid) == 0:
            continue
        for _, row in valid.iterrows():
            nn = int(row["nneighbours"])
            if nn < 3 or nn > max_nn:
                continue
            h0 = row["h0_dist"]
            if not isinstance(h0, (list, np.ndarray)) or len(h0) == 0:
                continue
            h0_arr = np.array(h0, dtype=np.float64)
            h, _ = np.histogram(h0_arr, bins=h0_bins)
            entropy_map[nn - 1] += h.astype(np.float32)

        if entropy_map.sum() < 1.0:
            continue

        last_nonzero = (entropy_map.sum(axis=1) > 0).nonzero()[0]
        if len(last_nonzero) == 0:
            continue
        t = int(last_nonzero[-1]) + 1
        trimmed = entropy_map[:t]
        row_sums = trimmed.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        trimmed = trimmed / row_sums

        maps.append(trimmed.astype(np.float32))
        metadata.append({
            "tag_name": tag_name,
            "doc_name": doc_name,
            "pnum": pnum,
            "query_granularity": query_gran,
            "corpus_granularity": corpus_gran,
            "n_tagged": gdf["n_tagged"].iloc[0] if "n_tagged" in gdf.columns else None,
            "t_effective": t,
        })

    return maps, metadata


def run_decoder_inference(model, maps: list[np.ndarray], device) -> np.ndarray:
    """
    Run inference on variable-length experimental maps.

    Returns (N, 50, 64) int8 with values in {-1, 0, +1}, padded with zeros for
    rows beyond each input's T_in.
    """
    from tqdm import tqdm
    model.eval()
    out = np.zeros((len(maps), MAX_NEIGHBORS, MAX_FEATURES), dtype=np.int8)
    with torch.no_grad():
        for i, m in enumerate(tqdm(maps, desc="Running inference")):
            x = torch.from_numpy(m[np.newaxis, :, :]).to(device)        # (1, T, 20)
            t = x.size(1)
            valid = torch.ones(1, t, dtype=torch.bool, device=device)
            class_logits = model(x, valid_mask=valid)[0]                # (T, F, 3)
            pred = (class_logits.argmax(dim=-1) - 1).cpu().numpy().astype(np.int8)
            out[i, :t, :] = pred
    return out


def aggregate_predictions(predictions: np.ndarray, metadata: list[dict], output_dir: Path):
    import pandas as pd
    meta_df = pd.DataFrame(metadata)
    n_shared = (predictions == 1).sum(axis=2)
    n_nonquery = (predictions == -1).sum(axis=2)
    n_absent = (predictions == 0).sum(axis=2)

    meta_df["mean_shared"] = n_shared.mean(axis=1)
    meta_df["mean_nonquery"] = n_nonquery.mean(axis=1)
    meta_df["mean_absent"] = n_absent.mean(axis=1)
    meta_df["shared_at_closest"] = n_shared[:, 0]
    meta_df["shared_at_farthest"] = n_shared[:, -1]

    print("\n" + "=" * 70)
    print("PREDICTION SUMMARY (v3)")
    print("=" * 70)
    print(f"Total samples: {len(meta_df)}")
    print(f"Mean shared features per neighbor: {meta_df['mean_shared'].mean():.2f}")
    print(f"Mean non-query features per neighbor: {meta_df['mean_nonquery'].mean():.2f}")

    if "corpus_granularity" in meta_df.columns:
        agg = (
            meta_df.groupby("corpus_granularity")
            .agg(
                n=("mean_shared", "count"),
                mean_shared=("mean_shared", "mean"),
                mean_nonquery=("mean_nonquery", "mean"),
                shared_closest=("shared_at_closest", "mean"),
                shared_farthest=("shared_at_farthest", "mean"),
            )
            .round(2)
        )
        print("\n--- By corpus_granularity ---")
        print(agg.to_string())

    meta_df.to_csv(output_dir / "prediction_summary.csv", index=False)
    print(f"\nSaved prediction_summary.csv to {output_dir}")


def run_prediction(args, device, output_dir):
    import pickle

    model_path = output_dir / "model_v3.pt"
    if args.model_from_s3:
        download_model_from_s3(model_path)
    elif not model_path.exists():
        print(f"No trained model at {model_path}. Train first or pass --model-from-s3.")
        sys.exit(1)

    model = FeaturePredictorV2(
        d_model=args.d_model, enc_layers=args.enc_layers,
        heads=args.heads, d_ff=args.d_ff, dropout=args.dropout,
    ).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))

    exp_maps, exp_metadata = build_experimental_entropy_maps(args.predict)
    print(f"Built {len(exp_maps)} experimental entropy maps "
          f"(T in [{min(len(m) for m in exp_maps)}, {max(len(m) for m in exp_maps)}])")

    predictions = run_decoder_inference(model, exp_maps, device)

    results = {
        "predictions": predictions,
        "metadata": exp_metadata,
        "entropy_maps": exp_maps,
    }
    results_path = output_dir / "predictions.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved predictions to {results_path}")

    aggregate_predictions(predictions, exp_metadata, output_dir)


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device()
    print(f"Using device: {device}")

    data_dir = resolve_data_dir(args.data_dir)

    if args.predict:
        run_prediction(args, device, output_dir)
        sys.exit(0)

    from torch.utils.data import DataLoader, Subset
    from torch.utils.tensorboard import SummaryWriter

    dataset = MapLabelDataset(
        str(data_dir / "entropy_maps.npy"),
        str(data_dir / "label_matrices.npy"),
        min_t_star=args.min_t_star,
    )
    print(f"Loaded dataset: {len(dataset)} samples")

    n_val = int(0.2 * len(dataset))
    rng = np.random.default_rng(42)
    p_feat_all = dataset.p_feat.numpy()
    quantiles = np.quantile(p_feat_all, [0.2, 0.4, 0.6, 0.8])
    strata = np.digitize(p_feat_all, quantiles)
    train_idx_parts, val_idx_parts = [], []
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        rng.shuffle(idx)
        n_val_s = int(round(0.2 * len(idx)))
        val_idx_parts.append(idx[:n_val_s])
        train_idx_parts.append(idx[n_val_s:])
    train_idx = np.concatenate(train_idx_parts)
    val_idx = np.concatenate(val_idx_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    train_ds = Subset(dataset, train_idx.tolist())
    val_ds = Subset(dataset, val_idx.tolist())
    n_train, n_val = len(train_ds), len(val_ds)

    if args.class_weight_strategy == "auto":
        args.class_weights = compute_inverse_frequency_weights(dataset, train_idx)
    else:
        args.class_weights = torch.tensor([1.0, 1.0, 1.0])
    print(f"Class weights (labels -1, 0, +1): {args.class_weights.tolist()}")

    use_cuda = device.type == "cuda"
    num_workers = max(0, args.num_workers)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        pin_memory=use_cuda,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model = FeaturePredictorV2(
        d_model=args.d_model, enc_layers=args.enc_layers,
        heads=args.heads, d_ff=args.d_ff, dropout=args.dropout,
    ).to(device)
    if args.compile:
        model = torch.compile(model)
    use_amp = use_cuda and not args.no_amp
    scaler = torch.amp.GradScaler("cuda") if (use_amp and args.amp_dtype == "fp16") else None
    if use_cuda:
        torch.backends.cudnn.benchmark = True
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training: {n_train} samples, Validation: {n_val} samples")
    print(f"AMP: {use_amp} (dtype={args.amp_dtype}) | compile: {args.compile} | num_workers: {num_workers}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = build_lr_scheduler(
        optimizer, total_steps=total_steps, warmup_steps=args.warmup_steps,
        peak_lr=args.lr, final_lr=1e-5,
    )

    tb_dir = output_dir / "runs"
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"TensorBoard logs: {tb_dir}")

    history = {
        "train_loss": [], "val_loss": [],
        "train_cell_acc": [], "val_cell_acc": [],
        "train_pos_recall": [], "val_pos_recall": [],
        "train_neg_recall": [], "val_neg_recall": [],
    }
    best_val = float("inf")
    patience = 0

    tb_upload_lock = threading.Lock()
    tb_upload_state = {"busy": False}

    def _tb_upload_bg():
        try:
            upload_dir_to_s3(tb_dir, S3_BUCKET, S3_TB_PREFIX)
        finally:
            with tb_upload_lock:
                tb_upload_state["busy"] = False

    for epoch in range(args.epochs):
        tr = train_one_epoch(model, train_loader, optimizer, scheduler, device, args,
                             scaler=scaler, use_amp=use_amp)
        ev = evaluate(model, val_loader, device, args, use_amp=use_amp)

        for k in ("loss", "cell_acc", "pos_recall", "neg_recall"):
            history[f"train_{k}"].append(tr[k])
            history[f"val_{k}"].append(ev[k])

        writer.add_scalars("Loss", {"train": tr["loss"], "val": ev["loss"]}, epoch)
        writer.add_scalars("CellAcc", {"train": tr["cell_acc"], "val": ev["cell_acc"]}, epoch)
        writer.add_scalars("PosRecall", {"train": tr["pos_recall"], "val": ev["pos_recall"]}, epoch)
        writer.add_scalars("NegRecall", {"train": tr["neg_recall"], "val": ev["neg_recall"]}, epoch)
        writer.add_scalars("TEff", {"train": tr["t_eff"], "val": ev["t_eff"]}, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
        writer.flush()
        if args.tb_upload_every > 0 and (epoch + 1) % args.tb_upload_every == 0:
            with tb_upload_lock:
                if not tb_upload_state["busy"]:
                    tb_upload_state["busy"] = True
                    threading.Thread(target=_tb_upload_bg, daemon=True).start()

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Loss: {tr['loss']:.4f}/{ev['loss']:.4f} | "
            f"Acc: {tr['cell_acc']:.4f}/{ev['cell_acc']:.4f} | "
            f"-1Rec: {tr['neg_recall']:.3f}/{ev['neg_recall']:.3f} | "
            f"+1Rec: {tr['pos_recall']:.3f}/{ev['pos_recall']:.3f} | "
            f"T_eff: {tr['t_eff']:.1f}/{ev['t_eff']:.1f}"
        )

        improved_meaningfully = ev["loss"] < best_val - args.early_stop_min_delta
        regressed = ev["loss"] > best_val
        if ev["loss"] < best_val:
            best_val = ev["loss"]
            sd = model.state_dict()
            if args.compile:
                sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
            torch.save(sd, output_dir / "model_v3.pt")
        if improved_meaningfully:
            patience = 0
        elif regressed:
            patience += 1
            if patience >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    writer.close()
    plot_training_history(history, output_dir)

    best_sd = torch.load(output_dir / "model_v3.pt", weights_only=True)
    target = model._orig_mod if args.compile and hasattr(model, "_orig_mod") else model
    target.load_state_dict(best_sd)
    ev = evaluate(model, val_loader, device, args, use_amp=use_amp)
    print(f"\nBest model -- loss: {ev['loss']:.4f}, "
          f"acc: {ev['cell_acc']:.4f}, "
          f"-1 recall: {ev['neg_recall']:.3f}, +1 recall: {ev['pos_recall']:.3f}")

    plot_reconstruction_comparison(model, dataset, device, output_dir, indices=val_idx)

    if args.upload_model_s3:
        upload_model_to_s3(output_dir / "model_v3.pt")
        upload_dir_to_s3(tb_dir, S3_BUCKET, S3_TB_PREFIX)
        s3 = _get_s3_client()
        for png in output_dir.glob("*.png"):
            key = f"{S3_PLOTS_PREFIX}/{png.name}"
            s3.upload_file(str(png), S3_BUCKET, key)
        print(f"Uploaded plots to s3://{S3_BUCKET}/{S3_PLOTS_PREFIX}/")
