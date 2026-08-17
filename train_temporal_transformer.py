import argparse
import glob
import math
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_DIR = "/data/projects/tongue_segmentation/code_represent/testvedio/muler_vedio"
DEFAULT_FEATURE_DIR = os.path.join(ROOT_DIR, "shap_results", "feature_sequences")
DEFAULT_CHECKPOINT = os.path.join(ROOT_DIR, "checkpoints", "temporal_transformer_best.pth")
DEFAULT_PREDICTION_DIR = os.path.join(ROOT_DIR, "shap_results", "temporal_predictions")
META_COLUMNS = {"frame_idx", "event_id", "source_group", "source_file"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_feature_sequences(feature_dir, csv_dir):
    files = list_feature_files(feature_dir)
    if files:
        return files

    print("Feature sequences not found. Generating them via SHAP.load_and_extract ...")
    from SHAP import load_and_extract

    load_and_extract(csv_dir)
    files = list_feature_files(feature_dir)
    if not files:
        raise RuntimeError("No feature sequence CSV files were generated.")
    return files


def list_feature_files(feature_dir):
    files = sorted(glob.glob(os.path.join(feature_dir, "*_features.csv")))
    return [path for path in files if not path.endswith("all_frame_features.csv")]


def infer_feature_columns(feature_files):
    common = None
    ordered = None
    for path in feature_files:
        cols = pd.read_csv(path, nrows=1).columns.tolist()
        cols = [col for col in cols if col not in META_COLUMNS]
        if ordered is None:
            ordered = cols
        if common is None:
            common = set(cols)
        else:
            common &= set(cols)

    if not ordered or not common:
        raise RuntimeError("Unable to infer feature columns from feature sequence CSV files.")

    feature_cols = [col for col in ordered if col in common]
    if not feature_cols:
        raise RuntimeError("No usable feature columns found.")
    return feature_cols


def split_feature_files(feature_files, val_fraction, seed):
    if len(feature_files) < 2:
        raise RuntimeError("At least two feature files are required for a train/val split.")

    shuffled = feature_files[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_fraction)))
    val_count = min(val_count, len(shuffled) - 1)
    val_files = shuffled[:val_count]
    train_files = shuffled[val_count:]
    return train_files, val_files


def build_onset_mask(active):
    previous = np.concatenate([np.zeros(1, dtype=np.int64), active[:-1]])
    return ((active == 1) & (previous == 0)).astype(np.float32)


def sanitize_feature_matrix(df, feature_cols):
    values = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return values.to_numpy(dtype=np.float32)


def build_windows_from_dataframe(
    df,
    feature_cols,
    scaler,
    seq_len_frames,
    stride_frames,
    horizon_frames,
    skip_active_end=True,
    max_windows=None,
):
    df = df.sort_values("frame_idx").reset_index(drop=True).copy()
    values = sanitize_feature_matrix(df, feature_cols)
    values = scaler.transform(values).astype(np.float32)

    if "event_id" in df.columns:
        active = (df["event_id"].fillna(0).to_numpy() > 0).astype(np.int64)
    else:
        active = np.zeros(len(df), dtype=np.int64)
    onset = build_onset_mask(active)

    max_horizon = max(horizon_frames)
    sequences = []
    labels = []
    meta = []

    for end in range(seq_len_frames - 1, len(df) - max_horizon, stride_frames):
        if skip_active_end and active[end] == 1:
            continue

        start = end - seq_len_frames + 1
        future_targets = []
        for horizon in horizon_frames:
            future_targets.append(float(onset[end + 1:end + horizon + 1].max()))

        sequences.append(values[start:end + 1])
        labels.append(future_targets)
        meta.append({
            "frame_idx": int(df.loc[end, "frame_idx"]),
            "event_id": int(df.loc[end, "event_id"]) if "event_id" in df.columns else 0,
        })

        if max_windows is not None and len(sequences) >= max_windows:
            break

    if not sequences:
        return np.empty((0, seq_len_frames, len(feature_cols)), dtype=np.float32), np.empty((0, len(horizon_frames)), dtype=np.float32), []

    return np.stack(sequences), np.asarray(labels, dtype=np.float32), meta


class TemporalSequenceDataset(Dataset):
    def __init__(self, sequences, labels, augment=False, noise_std=0.0, feature_mask_prob=0.0, scale_std=0.0):
        self.sequences = torch.as_tensor(sequences, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        self.augment = augment
        self.noise_std = noise_std
        self.feature_mask_prob = feature_mask_prob
        self.scale_std = scale_std

    def augment_sequence(self, sequence):
        augmented = sequence.clone()
        if self.scale_std > 0:
            scale = 1.0 + torch.randn((1, augmented.shape[-1]), dtype=augmented.dtype) * self.scale_std
            augmented = augmented * scale
        if self.noise_std > 0:
            augmented = augmented + torch.randn_like(augmented) * self.noise_std
        if self.feature_mask_prob > 0:
            mask = torch.rand_like(augmented) < self.feature_mask_prob
            augmented = augmented.masked_fill(mask, 0.0)
        return augmented

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        if self.augment:
            sequence = self.augment_sequence(sequence)
        return sequence, self.labels[idx]


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TemporalRiskTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        dropout,
        num_horizons,
        max_len,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len + 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, num_horizons)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.input_proj(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        pooled = self.norm(x[:, 0])
        pooled = self.dropout(pooled)
        return self.head(pooled)


def build_dataloaders(args, seq_len_frames, stride_frames, horizon_frames):
    feature_files = ensure_feature_sequences(args.feature_dir, args.csv_dir)
    if args.max_files is not None:
        feature_files = feature_files[:args.max_files]

    train_files, val_files = split_feature_files(feature_files, args.val_fraction, args.seed)
    feature_cols = infer_feature_columns(feature_files)

    train_frames = []
    for path in train_files:
        df = pd.read_csv(path)
        train_frames.append(sanitize_feature_matrix(df, feature_cols))

    scaler = StandardScaler()
    scaler.fit(np.concatenate(train_frames, axis=0))

    train_sequences = []
    train_labels = []
    val_sequences = []
    val_labels = []

    for path in train_files:
        df = pd.read_csv(path)
        sequences, labels, _ = build_windows_from_dataframe(
            df,
            feature_cols,
            scaler,
            seq_len_frames,
            stride_frames,
            horizon_frames,
            skip_active_end=not args.include_active_end,
            max_windows=args.max_windows_per_file,
        )
        if len(sequences):
            train_sequences.append(sequences)
            train_labels.append(labels)

    for path in val_files:
        df = pd.read_csv(path)
        sequences, labels, _ = build_windows_from_dataframe(
            df,
            feature_cols,
            scaler,
            seq_len_frames,
            stride_frames,
            horizon_frames,
            skip_active_end=not args.include_active_end,
            max_windows=args.max_windows_per_file,
        )
        if len(sequences):
            val_sequences.append(sequences)
            val_labels.append(labels)

    if not train_sequences or not val_sequences:
        raise RuntimeError("Unable to build non-empty train/val windows. Adjust sequence or horizon settings.")

    train_x = np.concatenate(train_sequences, axis=0)
    train_y = np.concatenate(train_labels, axis=0)
    val_x = np.concatenate(val_sequences, axis=0)
    val_y = np.concatenate(val_labels, axis=0)

    train_dataset = TemporalSequenceDataset(
        train_x,
        train_y,
        augment=True,
        noise_std=args.feature_noise_std,
        feature_mask_prob=args.feature_mask_prob,
        scale_std=args.feature_scale_std,
    )
    val_dataset = TemporalSequenceDataset(val_x, val_y)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    return train_loader, val_loader, feature_cols, scaler, train_y, train_files, val_files


def compute_metrics(targets, probs, horizon_names):
    metrics = {}
    aucs = []
    aps = []
    for idx, horizon_name in enumerate(horizon_names):
        y_true = targets[:, idx]
        y_prob = probs[:, idx]

        auc = float("nan")
        ap = float("nan")
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)
            aucs.append(auc)
            aps.append(ap)

        metrics[f"auc_{horizon_name}"] = auc
        metrics[f"ap_{horizon_name}"] = ap

    metrics["mean_auc"] = float(np.mean(aucs)) if aucs else float("nan")
    metrics["mean_ap"] = float(np.mean(aps)) if aps else float("nan")
    return metrics


def run_epoch(model, loader, criterion, optimizer, device, grad_clip_norm=None):
    model.train()
    total_loss = 0.0
    total_items = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        total_items += batch_x.size(0)

    return total_loss / max(total_items, 1)


def evaluate(model, loader, criterion, device, horizon_names):
    model.eval()
    total_loss = 0.0
    total_items = 0
    all_targets = []
    all_probs = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            total_items += batch_x.size(0)
            all_targets.append(batch_y.cpu().numpy())
            all_probs.append(torch.sigmoid(logits).cpu().numpy())

    targets = np.concatenate(all_targets, axis=0)
    probs = np.concatenate(all_probs, axis=0)
    metrics = compute_metrics(targets, probs, horizon_names)
    metrics["loss"] = total_loss / max(total_items, 1)
    return metrics


def save_training_history(history, output_dir):
    if not history:
        return

    os.makedirs(output_dir, exist_ok=True)
    history_df = pd.DataFrame(history)
    csv_path = os.path.join(output_dir, "training_history.csv")
    history_df.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train_loss", color="steelblue")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val_loss", color="darkred")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.2)
    axes[0].legend(loc="upper right")

    axes[1].plot(history_df["epoch"], history_df["lr"], label="lr", color="darkgreen")
    if "mean_ap" in history_df.columns:
        axes[1].plot(history_df["epoch"], history_df["mean_ap"], label="mean_ap", color="darkorange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Value")
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"), dpi=150)
    plt.close()


def save_checkpoint(path, model, feature_cols, scaler, args, seq_len_frames, stride_frames, horizon_frames, train_files, val_files):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_columns": feature_cols,
            "scaler_mean": scaler.mean_.astype(np.float32),
            "scaler_scale": scaler.scale_.astype(np.float32),
            "seq_len_frames": seq_len_frames,
            "stride_frames": stride_frames,
            "fps": args.fps,
            "horizons_sec": args.horizons_sec,
            "horizon_frames": horizon_frames,
            "model_kwargs": {
                "input_dim": len(feature_cols),
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_layers": args.num_layers,
                "dim_feedforward": args.dim_feedforward,
                "dropout": args.dropout,
                "num_horizons": len(horizon_frames),
                "max_len": seq_len_frames,
            },
            "train_files": train_files,
            "val_files": val_files,
        },
        path,
    )


def load_model_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = TemporalRiskTransformer(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


def compute_future_onset_targets(active, horizon_frames):
    active = np.asarray(active, dtype=np.int64)
    onset = build_onset_mask(active)
    targets = np.zeros((len(active), len(horizon_frames)), dtype=np.float32)

    for frame_idx in range(len(active)):
        for horizon_idx, horizon in enumerate(horizon_frames):
            hi = min(len(active), frame_idx + horizon + 1)
            if frame_idx + 1 < hi:
                targets[frame_idx, horizon_idx] = onset[frame_idx + 1:hi].max()

    return onset.astype(np.int64), targets


def predict_risk_dataframe(df, model, checkpoint, device):
    df = df.sort_values("frame_idx").reset_index(drop=True).copy()
    feature_cols = checkpoint["feature_columns"]
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    values = sanitize_feature_matrix(df, feature_cols)
    mean = checkpoint["scaler_mean"]
    scale = checkpoint["scaler_scale"]
    values = ((values - mean) / scale).astype(np.float32)

    seq_len_frames = int(checkpoint["seq_len_frames"])
    horizon_frames = checkpoint["horizon_frames"]
    horizon_names = [f"{sec:g}s" for sec in checkpoint["horizons_sec"]]

    if "event_id" in df.columns:
        active = (df["event_id"].fillna(0).to_numpy() > 0).astype(np.int64)
    else:
        active = np.zeros(len(df), dtype=np.int64)
    onset, targets = compute_future_onset_targets(active, horizon_frames)

    probs = np.full((len(df), len(horizon_frames)), np.nan, dtype=np.float32)
    with torch.no_grad():
        for end in range(seq_len_frames - 1, len(df)):
            seq = values[end - seq_len_frames + 1:end + 1]
            x = torch.from_numpy(seq).unsqueeze(0).to(device)
            logits = model(x)
            probs[end] = torch.sigmoid(logits)[0].cpu().numpy()

    out_df = pd.DataFrame({"frame_idx": df["frame_idx"].values})
    if "event_id" in df.columns:
        out_df["event_id"] = df["event_id"].values
    out_df["active"] = active
    out_df["onset"] = onset
    for idx, horizon_name in enumerate(horizon_names):
        out_df[f"target_{horizon_name}"] = targets[:, idx]
        out_df[f"risk_{horizon_name}"] = probs[:, idx]

    return out_df, horizon_names, horizon_frames


def extract_segments(binary_array):
    binary_array = np.asarray(binary_array, dtype=np.int64)
    segments = []
    start = None
    for idx, value in enumerate(binary_array):
        if value == 1 and start is None:
            start = idx
        elif value == 0 and start is not None:
            segments.append((start, idx - 1))
            start = None
    if start is not None:
        segments.append((start, len(binary_array) - 1))
    return segments


def summarize_prediction_dataframe(out_df, horizon_names, horizon_frames, fps, risk_threshold):
    frames = out_df["frame_idx"].to_numpy()
    active = out_df["active"].to_numpy(dtype=np.int64) if "active" in out_df.columns else np.zeros(len(out_df), dtype=np.int64)
    onset_idx = np.where(out_df["onset"].to_numpy(dtype=np.int64) == 1)[0] if "onset" in out_df.columns else np.array([], dtype=np.int64)
    duration_minutes = max(len(out_df) / max(fps, 1e-8) / 60.0, 1e-8)

    rows = []
    frame_payload = {}
    for horizon_idx, horizon_name in enumerate(horizon_names):
        risk_col = f"risk_{horizon_name}"
        target_col = f"target_{horizon_name}"
        risk = out_df[risk_col].to_numpy(dtype=np.float32)
        target = out_df[target_col].to_numpy(dtype=np.float32)
        valid = np.isfinite(risk)
        y_true = target[valid]
        y_prob = risk[valid]
        frame_payload[horizon_name] = (y_true, y_prob)

        auc = float("nan")
        ap = float("nan")
        if len(y_true) > 0 and len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)

        alert_binary = np.zeros(len(out_df), dtype=np.int64)
        alert_binary[valid] = (risk[valid] >= risk_threshold).astype(np.int64)
        alert_binary = alert_binary * (1 - active)
        alert_segments = extract_segments(alert_binary)

        true_alert_count = 0
        false_alert_count = 0
        for start_idx, _ in alert_segments:
            onset_found = False
            for onset_frame_idx in onset_idx:
                lead_frames = int(frames[onset_frame_idx] - frames[start_idx])
                if 0 < lead_frames <= horizon_frames[horizon_idx]:
                    onset_found = True
                    break
            if onset_found:
                true_alert_count += 1
            else:
                false_alert_count += 1

        lead_frames_list = []
        detected_onsets = 0
        for onset_frame_idx in onset_idx:
            lookback_lo = max(0, onset_frame_idx - horizon_frames[horizon_idx])
            lookback_hi = onset_frame_idx - 1
            candidates = []
            for seg_start, seg_end in alert_segments:
                if seg_end < lookback_lo or seg_start > lookback_hi:
                    continue
                candidates.append(max(seg_start, lookback_lo))
            if candidates:
                detected_onsets += 1
                earliest_alert_idx = min(candidates)
                lead_frames_list.append(int(frames[onset_frame_idx] - frames[earliest_alert_idx]))

        onset_count = int(len(onset_idx))
        event_recall = detected_onsets / onset_count if onset_count > 0 else float("nan")
        alert_precision = true_alert_count / len(alert_segments) if alert_segments else float("nan")
        mean_lead_frames = float(np.mean(lead_frames_list)) if lead_frames_list else float("nan")
        median_lead_frames = float(np.median(lead_frames_list)) if lead_frames_list else float("nan")
        max_lead_frames = float(np.max(lead_frames_list)) if lead_frames_list else float("nan")

        rows.append({
            "horizon": horizon_name,
            "frame_auc": auc,
            "frame_ap": ap,
            "valid_frame_count": int(valid.sum()),
            "onset_count": onset_count,
            "detected_onsets": int(detected_onsets),
            "event_recall": event_recall,
            "alert_count": int(len(alert_segments)),
            "true_alert_count": int(true_alert_count),
            "false_alert_count": int(false_alert_count),
            "alert_precision": alert_precision,
            "false_alerts_per_min": false_alert_count / duration_minutes,
            "mean_lead_frames": mean_lead_frames,
            "median_lead_frames": median_lead_frames,
            "max_lead_frames": max_lead_frames,
            "mean_lead_seconds": mean_lead_frames / fps if not math.isnan(mean_lead_frames) else float("nan"),
            "median_lead_seconds": median_lead_frames / fps if not math.isnan(median_lead_frames) else float("nan"),
            "max_lead_seconds": max_lead_frames / fps if not math.isnan(max_lead_frames) else float("nan"),
            "lead_frames_sum": float(np.sum(lead_frames_list)) if lead_frames_list else 0.0,
            "duration_minutes": duration_minutes,
        })

    return pd.DataFrame(rows), frame_payload


def save_prediction_outputs(out_df, horizon_names, out_csv, out_png, title):
    out_df.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    for horizon_name in horizon_names:
        ax.plot(out_df["frame_idx"], out_df[f"risk_{horizon_name}"], label=f"future {horizon_name}")

    if "active" in out_df.columns:
        active = out_df["active"].to_numpy(dtype=np.int64)
        active_segments = extract_segments(active)
        for start_idx, end_idx in active_segments:
            ax.axvspan(out_df.loc[start_idx, "frame_idx"], out_df.loc[end_idx, "frame_idx"], color="red", alpha=0.12)

    if "onset" in out_df.columns:
        onset_frames = out_df.loc[out_df["onset"] == 1, "frame_idx"].tolist()
        for onset_frame in onset_frames:
            ax.axvline(onset_frame, color="darkred", linestyle="--", linewidth=1.0, alpha=0.5)

    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Probability")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def export_onset_aligned_risk_reports(predictions_df, horizon_names, fps, output_dir, prefix, pre_seconds, post_seconds):
    if "onset" not in predictions_df.columns:
        return None, None

    pre_frames = max(1, int(round(pre_seconds * fps)))
    post_frames = max(1, int(round(post_seconds * fps)))
    aligned_rows = []

    for source_file, group_df in predictions_df.groupby("source_file"):
        group_df = group_df.sort_values("frame_idx").reset_index(drop=True)
        onset_positions = np.where(group_df["onset"].to_numpy(dtype=np.int64) == 1)[0]
        for event_number, onset_pos in enumerate(onset_positions, start=1):
            for rel_frame in range(-pre_frames, post_frames + 1):
                idx = onset_pos + rel_frame
                if idx < 0 or idx >= len(group_df):
                    continue

                row = {
                    "source_file": source_file,
                    "event_number": event_number,
                    "relative_frame": rel_frame,
                    "relative_seconds": rel_frame / fps,
                    "active": int(group_df.loc[idx, "active"]) if "active" in group_df.columns else 0,
                }
                for horizon_name in horizon_names:
                    row[f"risk_{horizon_name}"] = float(group_df.loc[idx, f"risk_{horizon_name}"])
                aligned_rows.append(row)

    if not aligned_rows:
        return None, None

    aligned_df = pd.DataFrame(aligned_rows)
    raw_csv = os.path.join(output_dir, f"{prefix}_onset_aligned_raw.csv")
    aligned_df.to_csv(raw_csv, index=False)

    summary_rows = []
    for horizon_name in horizon_names:
        grouped = aligned_df.groupby(["relative_frame", "relative_seconds"])
        summary = grouped.agg(
            mean_risk=(f"risk_{horizon_name}", "mean"),
            std_risk=(f"risk_{horizon_name}", "std"),
            event_count=(f"risk_{horizon_name}", "count"),
            mean_active=("active", "mean"),
        ).reset_index()
        summary.insert(0, "horizon", horizon_name)
        summary_rows.append(summary)

    summary_df = pd.concat(summary_rows, ignore_index=True)
    summary_df["std_risk"] = summary_df["std_risk"].fillna(0.0)
    summary_csv = os.path.join(output_dir, f"{prefix}_onset_aligned_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    fig, axes = plt.subplots(len(horizon_names), 1, figsize=(10, 3.2 * len(horizon_names)), sharex=True)
    if len(horizon_names) == 1:
        axes = [axes]

    for axis, horizon_name in zip(axes, horizon_names):
        subset = summary_df[summary_df["horizon"] == horizon_name].sort_values("relative_seconds")
        x = subset["relative_seconds"].to_numpy(dtype=np.float32)
        mean_risk = subset["mean_risk"].to_numpy(dtype=np.float32)
        std_risk = subset["std_risk"].to_numpy(dtype=np.float32)
        mean_active = subset["mean_active"].to_numpy(dtype=np.float32)

        axis.plot(x, mean_risk, color="steelblue", linewidth=2.0, label=f"risk_{horizon_name}")
        axis.fill_between(x, np.clip(mean_risk - std_risk, 0.0, 1.0), np.clip(mean_risk + std_risk, 0.0, 1.0), color="steelblue", alpha=0.18)
        axis.plot(x, mean_active, color="darkred", linestyle="--", linewidth=1.2, alpha=0.75, label="mean_active")
        axis.axvline(0.0, color="black", linestyle=":", linewidth=1.0)
        axis.set_ylabel("Probability")
        axis.set_ylim(0.0, 1.0)
        axis.set_title(f"Onset-aligned risk: {horizon_name}")
        axis.grid(alpha=0.2)
        axis.legend(loc="upper left")

    axes[-1].set_xlabel("Seconds relative to onset")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{prefix}_onset_aligned.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return summary_csv, plot_path


def export_validation_reports(args, val_files):
    device = torch.device(args.device)
    model, checkpoint = load_model_checkpoint(args.checkpoint_path, device)
    val_dir = os.path.join(args.prediction_out_dir, "validation")
    os.makedirs(val_dir, exist_ok=True)

    all_metrics = []
    all_predictions = []
    global_payload = {f"{sec:g}s": {"y_true": [], "y_prob": []} for sec in checkpoint["horizons_sec"]}

    for path in val_files:
        df = pd.read_csv(path)
        out_df, horizon_names, horizon_frames = predict_risk_dataframe(df, model, checkpoint, device)
        base = os.path.splitext(os.path.basename(path))[0]
        out_csv = os.path.join(val_dir, f"{base}_risk.csv")
        out_png = os.path.join(val_dir, f"{base}_risk.png")
        save_prediction_outputs(out_df, horizon_names, out_csv, out_png, f"Validation Risk Forecast - {base}")

        metrics_df, frame_payload = summarize_prediction_dataframe(out_df, horizon_names, horizon_frames, args.fps, args.risk_threshold)
        metrics_df.insert(0, "source_file", base)
        metrics_df.to_csv(os.path.join(val_dir, f"{base}_metrics.csv"), index=False)
        all_metrics.append(metrics_df)

        out_df.insert(0, "source_file", base)
        all_predictions.append(out_df)

        for horizon_name, (y_true, y_prob) in frame_payload.items():
            if len(y_true) > 0:
                global_payload[horizon_name]["y_true"].append(y_true)
                global_payload[horizon_name]["y_prob"].append(y_prob)

    if not all_metrics:
        print("No validation reports were generated.")
        return

    by_file_df = pd.concat(all_metrics, ignore_index=True)
    by_file_df.to_csv(os.path.join(val_dir, "validation_metrics_by_file.csv"), index=False)

    summary_rows = []
    for horizon_name in horizon_names:
        subset = by_file_df[by_file_df["horizon"] == horizon_name]
        auc = float("nan")
        ap = float("nan")
        if global_payload[horizon_name]["y_true"]:
            y_true = np.concatenate(global_payload[horizon_name]["y_true"], axis=0)
            y_prob = np.concatenate(global_payload[horizon_name]["y_prob"], axis=0)
            if len(np.unique(y_true)) > 1:
                auc = roc_auc_score(y_true, y_prob)
                ap = average_precision_score(y_true, y_prob)

        onset_count = int(subset["onset_count"].sum())
        detected_onsets = int(subset["detected_onsets"].sum())
        alert_count = int(subset["alert_count"].sum())
        true_alert_count = int(subset["true_alert_count"].sum())
        false_alert_count = int(subset["false_alert_count"].sum())
        duration_minutes = max(float(subset["duration_minutes"].sum()), 1e-8)
        lead_sum = float(subset["lead_frames_sum"].sum())

        summary_rows.append({
            "horizon": horizon_name,
            "frame_auc": auc,
            "frame_ap": ap,
            "onset_count": onset_count,
            "detected_onsets": detected_onsets,
            "event_recall": detected_onsets / onset_count if onset_count > 0 else float("nan"),
            "alert_count": alert_count,
            "true_alert_count": true_alert_count,
            "false_alert_count": false_alert_count,
            "alert_precision": true_alert_count / alert_count if alert_count > 0 else float("nan"),
            "false_alerts_per_min": false_alert_count / duration_minutes,
            "mean_lead_frames": lead_sum / detected_onsets if detected_onsets > 0 else float("nan"),
            "mean_lead_seconds": (lead_sum / detected_onsets) / args.fps if detected_onsets > 0 else float("nan"),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(val_dir, "validation_metrics_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    all_predictions_df.to_csv(os.path.join(val_dir, "validation_risk_predictions.csv"), index=False)

    onset_summary_csv, onset_plot_path = export_onset_aligned_risk_reports(
        all_predictions_df,
        horizon_names,
        args.fps,
        val_dir,
        "validation",
        args.pre_event_seconds,
        args.post_event_seconds,
    )

    print(f"Saved validation metrics summary to: {summary_csv}")
    if onset_summary_csv and onset_plot_path:
        print(f"Saved onset-aligned summary to: {onset_summary_csv}")
        print(f"Saved onset-aligned plot to: {onset_plot_path}")


def dataframe_from_prediction_input(csv_path):
    if csv_path.lower().endswith(".mp4"):
        raise ValueError(
            "--predict_csv expects a points_*.csv or *_features.csv file, not a video. "
            "Run test_video.py first to generate points_*.csv from the mp4."
        )

    df = pd.read_csv(csv_path)
    if "x0" in df.columns and "y0" in df.columns:
        from SHAP import add_temporal_features, extract_features_single_frame, load_points

        features = []
        for _, row in df.iterrows():
            pts = load_points(row)
            features.append(extract_features_single_frame(pts))
        df_feat = pd.DataFrame(features)
        df_feat = add_temporal_features(df_feat)
        df_feat.insert(0, "frame_idx", df["frame_idx"].values if "frame_idx" in df.columns else np.arange(len(df_feat)))
        df_feat.insert(1, "event_id", df["event_id"].values if "event_id" in df.columns else np.zeros(len(df_feat), dtype=int))
        return df_feat

    return df


def predict_single_file(args):
    device = torch.device(args.device)
    model, checkpoint = load_model_checkpoint(args.checkpoint_path, device)
    df = dataframe_from_prediction_input(args.predict_csv)
    out_df, horizon_names, horizon_frames = predict_risk_dataframe(df, model, checkpoint, device)

    os.makedirs(args.prediction_out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.predict_csv))[0]
    out_csv = os.path.join(args.prediction_out_dir, f"{base}_risk.csv")
    out_png = os.path.join(args.prediction_out_dir, f"{base}_risk.png")
    save_prediction_outputs(out_df, horizon_names, out_csv, out_png, "Temporal Transformer Risk Forecast")

    metrics_df, _ = summarize_prediction_dataframe(out_df, horizon_names, horizon_frames, args.fps, args.risk_threshold)
    metrics_csv = os.path.join(args.prediction_out_dir, f"{base}_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    single_predictions_df = out_df.copy()
    single_predictions_df.insert(0, "source_file", base)
    onset_summary_csv, onset_plot_path = export_onset_aligned_risk_reports(
        single_predictions_df,
        horizon_names,
        args.fps,
        args.prediction_out_dir,
        base,
        args.pre_event_seconds,
        args.post_event_seconds,
    )

    print(f"Saved risk curve CSV to: {out_csv}")
    print(f"Saved risk curve plot to: {out_png}")
    print(f"Saved event metrics CSV to: {metrics_csv}")
    if onset_summary_csv and onset_plot_path:
        print(f"Saved onset-aligned summary to: {onset_summary_csv}")
        print(f"Saved onset-aligned plot to: {onset_plot_path}")
    print(metrics_df.to_string(index=False))


def train(args):
    set_seed(args.seed)

    seq_len_frames = max(4, int(round(args.seq_seconds * args.fps)))
    stride_frames = max(1, int(round(args.stride_seconds * args.fps)))
    horizon_frames = [max(1, int(round(sec * args.fps))) for sec in args.horizons_sec]
    horizon_names = [f"{sec:g}s" for sec in args.horizons_sec]

    train_loader, val_loader, feature_cols, scaler, train_y, train_files, val_files = build_dataloaders(
        args, seq_len_frames, stride_frames, horizon_frames
    )

    positives = train_y.sum(axis=0)
    negatives = len(train_y) - positives
    pos_weight = np.where(positives > 0, negatives / np.maximum(positives, 1.0), 1.0)

    device = torch.device(args.device)
    model = TemporalRiskTransformer(
        input_dim=len(feature_cols),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        num_horizons=len(horizon_frames),
        max_len=seq_len_frames,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(pos_weight, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_decay_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )
    history = []
    epochs_without_improve = 0
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")
    print(f"Feature dim: {len(feature_cols)}, Seq len: {seq_len_frames} frames, Stride: {stride_frames} frames")
    print(f"Horizons: {args.horizons_sec}")
    print(f"Positive windows per horizon: {positives.astype(int).tolist()}")
    print(f"Trainable params: {trainable_params}")

    best_score = -float("inf")
    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, grad_clip_norm=args.grad_clip_norm)
        val_metrics = evaluate(model, val_loader, criterion, device, horizon_names)
        scheduler.step(val_metrics["loss"])
        score = val_metrics["mean_ap"] if not math.isnan(val_metrics["mean_ap"]) else -val_metrics["loss"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "mean_auc": val_metrics["mean_auc"],
                "mean_ap": val_metrics["mean_ap"],
                "lr": current_lr,
            }
        )

        metrics_text = ", ".join(
            [f"AP@{name}={val_metrics[f'ap_{name}']:.4f}" if not math.isnan(val_metrics[f'ap_{name}']) else f"AP@{name}=nan" for name in horizon_names]
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | mean_auc={val_metrics['mean_auc']:.4f} | "
            f"mean_ap={val_metrics['mean_ap']:.4f} | {metrics_text}"
        )

        if score > (best_score + args.min_improvement):
            best_score = score
            epochs_without_improve = 0
            save_checkpoint(
                args.checkpoint_path,
                model,
                feature_cols,
                scaler,
                args,
                seq_len_frames,
                stride_frames,
                horizon_frames,
                train_files,
                val_files,
            )
            print(f"Saved best checkpoint to: {args.checkpoint_path}")
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= args.early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch} after {epochs_without_improve} epochs without improvement.")
            break

    save_training_history(history, os.path.join(args.prediction_out_dir, "training"))
    export_validation_reports(args, val_files)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a temporal Transformer for future tongue-retroflexion risk.")
    parser.add_argument("--csv_dir", type=str, default=DEFAULT_CSV_DIR)
    parser.add_argument("--feature_dir", type=str, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--checkpoint_path", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--prediction_out_dir", type=str, default=DEFAULT_PREDICTION_DIR)
    parser.add_argument("--predict_csv", type=str, default=None)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--seq_seconds", type=float, default=4.0)
    parser.add_argument("--stride_seconds", type=float, default=0.2)
    parser.add_argument("--horizons_sec", type=float, nargs="+", default=[1.0, 3.0, 5.0])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--val_fraction", type=float, default=0.25)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--include_active_end", action="store_true")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_windows_per_file", type=int, default=None)
    parser.add_argument("--risk_threshold", type=float, default=0.5)
    parser.add_argument("--pre_event_seconds", type=float, default=5.0)
    parser.add_argument("--post_event_seconds", type=float, default=2.0)
    parser.add_argument("--feature_noise_std", type=float, default=0.02)
    parser.add_argument("--feature_mask_prob", type=float, default=0.05)
    parser.add_argument("--feature_scale_std", type=float, default=0.05)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--lr_patience", type=int, default=3)
    parser.add_argument("--lr_decay_factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--early_stop_patience", type=int, default=8)
    parser.add_argument("--min_improvement", type=float, default=1e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.predict_csv:
        predict_single_file(args)
    else:
        train(args)


if __name__ == "__main__":
    main()