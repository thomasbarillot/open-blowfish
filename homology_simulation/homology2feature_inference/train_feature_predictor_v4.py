#!/usr/bin/env python3
"""
Feature Predictor v4 — label-VAE pretraining + entropy-to-z alignment.

Two phases:
  - --phase vae:   train LabelEncoder + LabelDecoder on labels alone (ELBO).
  - --phase align: freeze label VAE, train EntropyEncoder so that its z_pred
                   matches the frozen LabelEncoder's mu, plus an auxiliary CE
                   on labels reconstructed through the frozen LabelDecoder.

Inference: entropy -> EntropyEncoder -> z -> LabelDecoder -> argmax labels.

Reuses RowEncoder, RowTransformerEncoder, MapLabelDataset, class_loss_and_acc,
compute_inverse_frequency_weights, plot_reconstruction_comparison from v3
(scripts/ottp_topology_analysis/train_feature_predictor_v2.py).
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
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_feature_predictor_v2 as v3  # baseline; we import, never mutate

S3_BUCKET = "homology-experiment"
S3_PREFIX = "simulation_dataset_v2"
S3_TB_PREFIX = "model/runs_v4"
S3_PLOTS_PREFIX = "model/plots_v4"

MAX_NEIGHBORS = v3.MAX_NEIGHBORS
MAX_FEATURES = v3.MAX_FEATURES
N_H0_BINS = v3.N_H0_BINS
N_CLASSES = v3.N_CLASSES

DEFAULT_D_MODEL = 384
DEFAULT_ENC_LAYERS = 4
DEFAULT_HEADS = 6
DEFAULT_D_FF = 1536
DEFAULT_DROPOUT = 0.1
DEFAULT_D_Z = 128

NUM_POS_FEATURES = 10  # divisor for the faithfulness target (#+1 / 10)



def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Mean KL[N(mu, sigma) || N(0, I)] across the batch, summed over latent dims.
    """
    per_sample = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)
    return per_sample.mean()


def kl_with_free_bits(mu: torch.Tensor, logvar: torch.Tensor,
                      free_bits: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-dim free-bits clamping: each latent dim is allowed to drop to `free_bits`
    nats of KL "for free" (no gradient pressure toward the prior below that floor).

    Returns (kl_for_loss, kl_raw) where kl_raw is the standard mean KL used for
    logging/diagnostics, and kl_for_loss is the clamped quantity used in the ELBO.
    """
    # KL per (batch, latent dim), then mean across batch -> per-dim KL.
    per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())  # (B, D)
    per_dim_mean = per_dim.mean(dim=0)                           # (D,)
    kl_raw = per_dim_mean.sum()
    if free_bits <= 0.0:
        return kl_raw, kl_raw
    kl_clamped = torch.clamp(per_dim_mean, min=free_bits).sum()
    return kl_clamped, kl_raw


class EntropyEncoder(nn.Module):
    """
    Deterministic encoder: entropy map -> point estimate of z.

    Reuses v3.RowEncoder + v3.RowTransformerEncoder, then masked-mean over T,
    then Linear projection to d_z.
    """

    def __init__(
        self,
        d_model: int = DEFAULT_D_MODEL,
        enc_layers: int = DEFAULT_ENC_LAYERS,
        heads: int = DEFAULT_HEADS,
        d_ff: int = DEFAULT_D_FF,
        dropout: float = DEFAULT_DROPOUT,
        d_z: int = DEFAULT_D_Z,
        seq_len: int = MAX_NEIGHBORS,
        n_features: int = MAX_FEATURES,
        n_h0_bins: int = N_H0_BINS,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.row_enc = v3.RowEncoder(d_model=d_model, n_h0_bins=n_h0_bins)
        self.register_buffer(
            "pos_enc",
            v3.sinusoidal_positional_encoding(seq_len, d_model),
            persistent=False,
        )
        self.row_attn = v3.RowTransformerEncoder(
            d_model=d_model, n_heads=heads, n_layers=enc_layers,
            d_ff=d_ff, dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_z),
        )

    def forward(self, entropy_map: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        t = entropy_map.size(1)
        x = self.row_enc(entropy_map) + self.pos_enc[:t].unsqueeze(0)
        kpm = ~valid_mask
        empty = ~valid_mask.any(dim=1)
        if empty.any():
            kpm = kpm.clone()
            kpm[empty, 0] = False
        h = self.row_attn(x, key_padding_mask=kpm)
        mask_f = valid_mask.to(h.dtype).unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        pooled = (h * mask_f).sum(dim=1) / denom
        return self.head(pooled)


class SummaryHead(nn.Module):
    """
    Decoder for the per-neighbor (noise, faithfulness) summary.

    Input:
      z: (B, d_z)
    Output:
      summary: (B, T, 2) in [0, 1] — channel 0 = noise[t], channel 1 = faith[t]

    Same shape pattern as LabelDecoder (broadcast z across T, add a learned
    per-position embedding, attend, project to 2). Sigmoid keeps outputs in
    [0, 1] which matches the natural range of both channels.
    """

    def __init__(
        self,
        d_model: int = DEFAULT_D_MODEL,
        enc_layers: int = DEFAULT_ENC_LAYERS,
        heads: int = DEFAULT_HEADS,
        d_ff: int = DEFAULT_D_FF,
        dropout: float = DEFAULT_DROPOUT,
        d_z: int = DEFAULT_D_Z,
        seq_len: int = MAX_NEIGHBORS,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.z_proj = nn.Linear(d_z, d_model)
        self.head_pos_emb = nn.Parameter(torch.randn(seq_len, d_model) * 0.02)
        self.row_attn = v3.RowTransformerEncoder(
            d_model=d_model, n_heads=heads, n_layers=enc_layers,
            d_ff=d_ff, dropout=dropout,
        )
        self.out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.size(0)
        h = self.z_proj(z).unsqueeze(1).expand(b, self.seq_len, -1)
        h = h + self.head_pos_emb.unsqueeze(0)
        h = self.row_attn(h)
        return torch.sigmoid(self.out(h))


def beta_at_epoch(epoch: int, total_epochs: int, target: float, warmup_frac: float) -> float:
    """
    Linear KL-weight warmup. beta = 0 at epoch 0, ramps linearly to `target`
    by epoch `warmup_frac * total_epochs`, stays at target after.
    """
    if warmup_frac <= 0.0:
        return target
    warmup_epochs = max(1, int(round(warmup_frac * total_epochs)))
    if epoch >= warmup_epochs:
        return target
    return target * (epoch / warmup_epochs)


def _move_label_batch(batch, device):
    maps, signed_targets = batch
    return (
        maps.to(device, non_blocking=True),
        signed_targets.to(device, non_blocking=True),
    )


def _render_recon_grid(all_recon, all_targets, all_maps, all_t_eff,
                       output_dir: Path, n_examples: int, title_recon: str):
    recon = np.concatenate(all_recon)
    targets = np.concatenate(all_targets)
    maps = np.concatenate(all_maps)
    t_eff_all = np.concatenate(all_t_eff)

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
        ax_rc.set_title(title_recon)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train v4 label-VAE / entropy-alignment feature predictor")
    parser.add_argument(
        "--phase", type=str, default="vae",
        choices=["vae", "align", "sanity", "direct", "summary"],
        help="vae=Phase 1; align=Phase 2; "
             "sanity=load VAE, encode->decode labels at posterior mean (no sampling, "
             "no entropy encoder) and report val recall; "
             "direct=train EntropyEncoder + LabelDecoder end-to-end on labels "
             "with weighted CE (no VAE in the loop); "
             "summary=train EntropyEncoder + SummaryHead end-to-end to predict "
             "per-neighbor (noise=#-1/64, faith=#+1/10) summaries (no VAE).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).parent / "outputs" / "simulation_dataset_v2"),
        help="Local dir or s3:// URI with entropy_maps.npy and label_matrices.npy",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).parent / "outputs" / "model_v4"),
    )
    parser.add_argument("--vae-checkpoint", type=str, default=None,
                        help="Path to a Phase 1 (label-VAE) checkpoint. Required for --phase align.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-3)
    parser.add_argument("--prefix-truncation", action="store_true")
    parser.add_argument("--min-prefix", type=int, default=3)
    parser.add_argument("--min-t-star", type=int, default=2)
    parser.add_argument("--class-weight-strategy", type=str, default="auto",
                        choices=["auto", "uniform"])
    parser.add_argument("--pos-weight-boost", type=float, default=1.0,
                        help="Multiplicative boost on the +1 class weight after auto inverse-freq "
                             "computation. Re-normalized to mean=1 so loss magnitude stays comparable. "
                             "Use values >1 (e.g. 4-8) when +1 recall stays stuck near 0.")
    # Phase 1 (vae)
    parser.add_argument("--d-z", type=int, default=DEFAULT_D_Z)
    parser.add_argument("--beta-target", type=float, default=0.05)
    parser.add_argument("--beta-warmup-frac", type=float, default=0.2)
    parser.add_argument("--free-bits", type=float, default=0.5,
                        help="Per-dim KL floor in nats. 0 disables. Prevents posterior "
                             "collapse: dims below the floor get no gradient pressure "
                             "from the KL term, so reconstruction can keep using them.")
    parser.add_argument("--vae-monitor", type=str, default="composite",
                        choices=["loss", "cell_acc", "composite"],
                        help="Validation metric used for best-checkpoint selection and "
                             "early stopping in Phase 1. 'composite' = cell_acc + 0.5*pos_recall. "
                             "loss=lower-is-better; the others are higher-is-better.")
    # Phase 2 (align)
    parser.add_argument("--lambda-aux-ce", type=float, default=0.1)
    parser.add_argument("--lambda-mse", type=float, default=1.0,
                        help="Weight on the MSE-to-mu term in the align loss. "
                             "Set to 0 for a pure aux-CE ablation (skips the "
                             "isotropic L2 pressure that ignores +1 recall).")
    # Model geometry
    parser.add_argument("--d-model", type=int, default=DEFAULT_D_MODEL)
    parser.add_argument("--enc-layers", type=int, default=DEFAULT_ENC_LAYERS)
    parser.add_argument("--heads", type=int, default=DEFAULT_HEADS)
    parser.add_argument("--d-ff", type=int, default=DEFAULT_D_FF)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    # Training infra
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--amp-dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--upload-model-s3", action="store_true")
    parser.add_argument("--tb-upload-every", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap dataset for quick iterations.")
    return parser.parse_args()


def _build_dataset_and_split(args, device):
    data_dir = v3.resolve_data_dir(args.data_dir)
    dataset = v3.MapLabelDataset(
        str(data_dir / "entropy_maps.npy"),
        str(data_dir / "label_matrices.npy"),
        min_t_star=args.min_t_star,
    )
    if args.max_samples is not None and args.max_samples < len(dataset):
        dataset_subset = Subset(dataset, list(range(args.max_samples)))
        ds_for_split = dataset_subset
        p_feat_all = None
    else:
        ds_for_split = dataset
        p_feat_all = dataset.p_feat.numpy()
    print(f"Loaded dataset: {len(ds_for_split)} samples")

    n = len(ds_for_split)
    rng = np.random.default_rng(42)
    if p_feat_all is not None:
        quantiles = np.quantile(p_feat_all, [0.2, 0.4, 0.6, 0.8])
        strata = np.digitize(p_feat_all, quantiles)
        train_parts, val_parts = [], []
        for s in np.unique(strata):
            idx = np.where(strata == s)[0]
            rng.shuffle(idx)
            n_val_s = int(round(0.2 * len(idx)))
            val_parts.append(idx[:n_val_s])
            train_parts.append(idx[n_val_s:])
        train_idx = np.concatenate(train_parts)
        val_idx = np.concatenate(val_parts)
    else:
        idx = np.arange(n)
        rng.shuffle(idx)
        n_val = int(0.2 * n)
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    if args.class_weight_strategy == "auto" and isinstance(ds_for_split, v3.MapLabelDataset):
        args.class_weights = v3.compute_inverse_frequency_weights(ds_for_split, train_idx)
    else:
        args.class_weights = torch.tensor([1.0, 1.0, 1.0])
    if getattr(args, "pos_weight_boost", 1.0) != 1.0:
        # Multiply the +1 weight by an extra factor on top of inverse-frequency,
        # then re-normalize to mean=1 so total loss magnitude stays comparable.
        # Used to push the optimizer harder on the rare positive class when
        # plain inverse-frequency leaves +1 recall stuck at ~0.
        boosted = args.class_weights.clone()
        boosted[2] = boosted[2] * args.pos_weight_boost
        boosted = boosted * (boosted.numel() / boosted.sum())
        args.class_weights = boosted
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
    train_loader = DataLoader(Subset(ds_for_split, train_idx.tolist()),
                              shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(Subset(ds_for_split, val_idx.tolist()),
                            shuffle=False, **loader_kwargs)
    return dataset, ds_for_split, train_idx, val_idx, train_loader, val_loader


def summary_targets_from_labels(target_signed: torch.Tensor) -> torch.Tensor:
    """
    Convert (B, T, F) signed labels in {-1, 0, +1} into (B, T, 2) summary
    targets. Channel 0 = #(-1)/F (noise), channel 1 = #(+1)/NUM_POS_FEATURES
    (faithfulness, target divisor=10 by spec).

    F is read from target_signed.shape[-1] for safety; both divisors are
    fixed-positive constants — calibration choices, not model hyperparams.
    """
    f = float(target_signed.shape[-1])
    neg = (target_signed == -1).sum(dim=-1).to(torch.float32) / f
    pos = (target_signed == 1).sum(dim=-1).to(torch.float32) / float(NUM_POS_FEATURES)
    return torch.stack([neg, pos], dim=-1)


def summary_loss_and_metrics(
    pred_summary: torch.Tensor,
    target_summary: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict:
    """
    Masked MSE on (B, T, 2) summary outputs. Reports total MSE plus per-channel
    MSE and Pearson correlation, since (noise, faith) live on different scales
    and averaging the two together hides per-channel behavior.

    valid_mask: (B, T) bool — only valid neighbors contribute.
    """
    mask = valid_mask.unsqueeze(-1).to(pred_summary.dtype)         # (B, T, 1)
    diff_sq = (pred_summary - target_summary) ** 2                  # (B, T, 2)
    denom = mask.sum().clamp(min=1.0) * 2.0
    mse = (diff_sq * mask).sum() / denom

    per_channel_denom = mask.sum().clamp(min=1.0)
    mse_noise = (diff_sq[..., 0:1] * mask).sum() / per_channel_denom
    mse_faith = (diff_sq[..., 1:2] * mask).sum() / per_channel_denom

    flat_mask = valid_mask.reshape(-1)
    p = pred_summary.reshape(-1, 2)[flat_mask]
    t = target_summary.reshape(-1, 2)[flat_mask]
    corr_noise = _pearson(p[:, 0], t[:, 0])
    corr_faith = _pearson(p[:, 1], t[:, 1])

    return {
        "loss": mse,
        "mse": mse.detach(),
        "mse_noise": mse_noise.detach(),
        "mse_faith": mse_faith.detach(),
        "corr_noise": corr_noise.detach(),
        "corr_faith": corr_faith.detach(),
    }


def _pearson(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pearson r between two 1D tensors. Returns 0 if either has no variance."""
    if x.numel() == 0:
        return torch.tensor(0.0, device=x.device)
    xm = x - x.mean()
    ym = y - y.mean()
    denom = (xm.pow(2).sum().sqrt() * ym.pow(2).sum().sqrt()).clamp(min=1e-8)
    return (xm * ym).sum() / denom


def train_one_epoch_summary(entropy_encoder, summary_head, loader, optimizer,
                            scheduler, device, args, scaler=None, use_amp=False):
    entropy_encoder.train()
    summary_head.train()
    running = {"loss": 0.0, "mse": 0.0, "mse_noise": 0.0, "mse_faith": 0.0,
               "corr_noise": 0.0, "corr_faith": 0.0, "t_eff": 0.0}
    n_samples = 0
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if use_amp else torch.autocast(device_type=device.type, enabled=False)
    )
    params = list(entropy_encoder.parameters()) + list(summary_head.parameters())

    for batch in loader:
        maps, signed_targets = _move_label_batch(batch, device)
        maps, signed_targets, mask, t_eff = v3._prepare_batch(
            maps, signed_targets,
            random_truncate=args.prefix_truncation, min_prefix=args.min_prefix,
        )
        target_summary = summary_targets_from_labels(signed_targets)
        optimizer.zero_grad()
        with amp_ctx:
            z = entropy_encoder(maps, mask)
            pred = summary_head(z)
            metrics = summary_loss_and_metrics(pred, target_summary, mask)
        loss = metrics["loss"]
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer_stepped = True
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_stepped = scaler.get_scale() >= scale_before
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
        if optimizer_stepped:
            scheduler.step()

        bs = maps.size(0)
        for k in ("loss", "mse", "mse_noise", "mse_faith", "corr_noise", "corr_faith"):
            running[k] += float(metrics[k]) * bs
        running["t_eff"] += float(t_eff.float().mean()) * bs
        n_samples += bs

    return {k: v / n_samples for k, v in running.items()}


def evaluate_summary(entropy_encoder, summary_head, loader, device, args, use_amp=False):
    entropy_encoder.eval()
    summary_head.eval()
    running = {"loss": 0.0, "mse": 0.0, "mse_noise": 0.0, "mse_faith": 0.0,
               "corr_noise": 0.0, "corr_faith": 0.0, "t_eff": 0.0}
    n_samples = 0
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if use_amp else torch.autocast(device_type=device.type, enabled=False)
    )
    with torch.no_grad():
        for batch in loader:
            maps, signed_targets = _move_label_batch(batch, device)
            maps, signed_targets, mask, t_eff = v3._prepare_batch(
                maps, signed_targets, random_truncate=False, min_prefix=args.min_prefix,
            )
            target_summary = summary_targets_from_labels(signed_targets)
            with amp_ctx:
                z = entropy_encoder(maps, mask)
                pred = summary_head(z)
                metrics = summary_loss_and_metrics(pred, target_summary, mask)
            bs = maps.size(0)
            for k in ("loss", "mse", "mse_noise", "mse_faith", "corr_noise", "corr_faith"):
                running[k] += float(metrics[k]) * bs
            running["t_eff"] += float(t_eff.float().mean()) * bs
            n_samples += bs
    return {k: v / n_samples for k, v in running.items()}


def plot_summary_history(history: dict, output_dir: Path):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    axes[0].plot(history["train_loss"], label="Train"); axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_title("MSE (mean)"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(history["train_mse_noise"], label="Train"); axes[1].plot(history["val_mse_noise"], label="Val")
    axes[1].set_title("MSE noise"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    axes[2].plot(history["train_mse_faith"], label="Train"); axes[2].plot(history["val_mse_faith"], label="Val")
    axes[2].set_title("MSE faithfulness"); axes[2].set_xlabel("Epoch"); axes[2].legend()
    axes[3].plot(history["train_corr_noise"], label="Train noise")
    axes[3].plot(history["val_corr_noise"], label="Val noise")
    axes[3].plot(history["train_corr_faith"], label="Train faith")
    axes[3].plot(history["val_corr_faith"], label="Val faith")
    axes[3].set_title("Pearson r"); axes[3].set_xlabel("Epoch"); axes[3].legend()
    axes[3].set_ylim(-0.1, 1.0)
    plt.tight_layout()
    fig.savefig(output_dir / "training_history.png", dpi=150)
    plt.close(fig)


def plot_summary_reconstruction(entropy_encoder, summary_head, dataset, device,
                                output_dir: Path, n_examples: int = 8, indices=None):
    """Side-by-side per-sample images: target (50, 2) vs predicted (50, 2)."""
    entropy_encoder.eval()
    summary_head.eval()
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

    all_pred, all_target, all_t_eff = [], [], []
    with torch.no_grad():
        for batch in loader:
            maps, signed_targets = _move_label_batch(batch, device)
            maps, signed_targets, mask, t_eff = v3._prepare_batch(
                maps, signed_targets, random_truncate=False, min_prefix=1,
            )
            target = summary_targets_from_labels(signed_targets)
            z = entropy_encoder(maps, mask)
            pred = summary_head(z)
            all_pred.append(pred.cpu().numpy())
            all_target.append(target.cpu().numpy())
            all_t_eff.append(t_eff.cpu().numpy())

    pred = np.concatenate(all_pred)
    target = np.concatenate(all_target)
    t_eff_all = np.concatenate(all_t_eff)

    per_sample_err = np.zeros(len(pred), dtype=np.float32)
    for i in range(len(pred)):
        te = int(t_eff_all[i])
        if te <= 0:
            per_sample_err[i] = np.inf
            continue
        per_sample_err[i] = float(np.mean((pred[i, :te] - target[i, :te]) ** 2))
    sorted_idx = np.argsort(per_sample_err)
    picks = np.linspace(0, len(sorted_idx) - 1, n_examples, dtype=int)
    pick_idxs = sorted_idx[picks]

    fig, axes = plt.subplots(n_examples, 2, figsize=(8, 3 * n_examples))
    if n_examples == 1:
        axes = axes[np.newaxis, :]
    for row, i in enumerate(pick_idxs):
        te = int(t_eff_all[i])
        ax_t = axes[row, 0]
        im = ax_t.imshow(target[i], cmap="viridis", aspect="auto", vmin=0, vmax=1)
        ax_t.axhline(te - 0.5, color="cyan", linestyle="--", linewidth=1.0)
        ax_t.set_title(f"Target (mse={per_sample_err[i]:.4f}, T*={te})")
        ax_t.set_xticks([0, 1]); ax_t.set_xticklabels(["noise", "faith"])
        ax_t.set_ylabel("neighbor")
        plt.colorbar(im, ax=ax_t)

        ax_p = axes[row, 1]
        im = ax_p.imshow(pred[i], cmap="viridis", aspect="auto", vmin=0, vmax=1)
        ax_p.axhline(te - 0.5, color="cyan", linestyle="--", linewidth=1.0)
        ax_p.set_title("Predicted")
        ax_p.set_xticks([0, 1]); ax_p.set_xticklabels(["noise", "faith"])
        plt.colorbar(im, ax=ax_p)

    plt.tight_layout()
    fig.savefig(output_dir / "reconstruction_comparison.png", dpi=150)
    plt.close(fig)


def run_summary(args, device, output_dir):
    """
    entropy -> z -> (noise[t], faith[t]) per neighbor.

    Replaces the (50, 64) label target with the (50, 2) summary
    [#(-1)/64, #(+1)/10] computed from the labels. Trains EntropyEncoder +
    SummaryHead jointly, no VAE in the loop.
    """
    dataset, _, _, val_idx, train_loader, val_loader = _build_dataset_and_split(args, device)

    enc_geom = dict(
        d_model=args.d_model, enc_layers=args.enc_layers, heads=args.heads,
        d_ff=args.d_ff, dropout=args.dropout, d_z=args.d_z,
    )
    head_geom = dict(enc_geom)
    entropy_encoder = EntropyEncoder(**enc_geom).to(device)
    summary_head = SummaryHead(**head_geom).to(device)
    if args.compile:
        entropy_encoder = torch.compile(entropy_encoder)
        summary_head = torch.compile(summary_head)
    use_amp = (device.type == "cuda") and not args.no_amp
    scaler = torch.amp.GradScaler("cuda") if (use_amp and args.amp_dtype == "fp16") else None
    n_params = sum(p.numel() for p in entropy_encoder.parameters()) + \
               sum(p.numel() for p in summary_head.parameters())
    print(f"Summary (encoder + head) parameters: {n_params:,}")

    params = list(entropy_encoder.parameters()) + list(summary_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = v3.build_lr_scheduler(
        optimizer, total_steps=total_steps, warmup_steps=args.warmup_steps,
        peak_lr=args.lr, final_lr=1e-5,
    )

    tb_dir = output_dir / "runs"
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"TensorBoard logs: {tb_dir}")

    history = {f"{split}_{k}": [] for split in ("train", "val")
               for k in ("loss", "mse", "mse_noise", "mse_faith", "corr_noise", "corr_faith")}
    best_val = float("inf")
    patience = 0

    for epoch in range(args.epochs):
        tr = train_one_epoch_summary(entropy_encoder, summary_head, train_loader,
                                     optimizer, scheduler, device, args,
                                     scaler=scaler, use_amp=use_amp)
        ev = evaluate_summary(entropy_encoder, summary_head, val_loader, device, args,
                              use_amp=use_amp)

        for k in ("loss", "mse", "mse_noise", "mse_faith", "corr_noise", "corr_faith"):
            history[f"train_{k}"].append(tr[k])
            history[f"val_{k}"].append(ev[k])
            writer.add_scalars(k, {"train": tr[k], "val": ev[k]}, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
        writer.flush()

        print(
            f"Summary Epoch {epoch+1:3d}/{args.epochs} | "
            f"Loss: {tr['loss']:.5f}/{ev['loss']:.5f} | "
            f"MSE_n: {tr['mse_noise']:.5f}/{ev['mse_noise']:.5f} | "
            f"MSE_f: {tr['mse_faith']:.5f}/{ev['mse_faith']:.5f} | "
            f"r_n: {tr['corr_noise']:.3f}/{ev['corr_noise']:.3f} | "
            f"r_f: {tr['corr_faith']:.3f}/{ev['corr_faith']:.3f}"
        )

        improved_meaningfully = ev["loss"] < best_val - args.early_stop_min_delta
        if ev["loss"] < best_val:
            best_val = ev["loss"]
            sd_e = entropy_encoder.state_dict()
            sd_h = summary_head.state_dict()
            if args.compile:
                sd_e = {k.removeprefix("_orig_mod."): v for k, v in sd_e.items()}
                sd_h = {k.removeprefix("_orig_mod."): v for k, v in sd_h.items()}
            torch.save({"encoder_state_dict": sd_e, "head_state_dict": sd_h,
                        "geom": enc_geom},
                       output_dir / "model_v4_summary.pt")
        if improved_meaningfully:
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    writer.close()
    plot_summary_history(history, output_dir)

    best_ckpt = torch.load(output_dir / "model_v4_summary.pt", weights_only=True,
                           map_location=device)
    target_enc = entropy_encoder._orig_mod if args.compile and hasattr(entropy_encoder, "_orig_mod") else entropy_encoder
    target_head = summary_head._orig_mod if args.compile and hasattr(summary_head, "_orig_mod") else summary_head
    target_enc.load_state_dict(best_ckpt["encoder_state_dict"])
    target_head.load_state_dict(best_ckpt["head_state_dict"])
    plot_summary_reconstruction(target_enc, target_head, dataset, device, output_dir,
                                indices=val_idx)


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = v3.resolve_device()
    print(f"Using device: {device}")
    run_summary(args, device, output_dir)
   