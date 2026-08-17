"""Prototype + temporal prior utilities for the Tongue task (方案一: 4 prototypes + 1 temporal).

This module builds and consumes the 5-channel ``cls_sim_avg_label_input`` prior:

  * channels 0-3: K=4 average-mask shape prototypes, per-frame weighted by the
    appearance similarity between the current frame and each cluster centroid (B).
  * channel 4:    the previous frame's mask (temporal prior, C). During training this
    is the previous labeled frame's GT mask (teacher forcing) with perturbation
    augmentation; during inference it is the previous frame's predicted mask.

Build the prototypes/centroids once with::

    python -m utils.tongue_prior --data_path <repo_root> --out_dir checkpoints/tongue_prior

The build reads ``<data_path>/dataset.json`` (COCO) + ``<data_path>/image``.
"""

import argparse
import json
import os

import cv2
import numpy as np

DEFAULT_PRIOR_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "tongue_prior")
PRIOR_FILE_NAME = "tongue_prior.npz"
N_CLUSTERS = 4
FEAT_SIZE = 16
DEFAULT_TEMPERATURE = 0.1


def _mask_from_annotations(anns, height, width):
    """Rasterize COCO polygon annotations into a binary (H, W) uint8 mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in anns:
        seg = ann.get("segmentation")
        if not seg:
            continue
        poly = np.array(seg[0]).reshape(-1, 2)
        cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
    return mask


def _appearance_feature(gray_image, feat_size=FEAT_SIZE):
    """Cheap, label-free appearance descriptor: downsampled + flattened grayscale."""
    small = cv2.resize(gray_image, (feat_size, feat_size), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32).reshape(-1) / 255.0


def build_priors(data_path, out_dir=DEFAULT_PRIOR_DIR, n_clusters=N_CLUSTERS,
                 feat_size=FEAT_SIZE, img_size=256, temperature=DEFAULT_TEMPERATURE, seed=42):
    """Cluster training frames by appearance and export shape prototypes + centroids."""
    from pycocotools.coco import COCO
    from sklearn.cluster import KMeans

    json_path = os.path.join(data_path, "dataset.json")
    img_dir = os.path.join(data_path, "image")
    coco = COCO(json_path)
    img_ids = list(coco.imgs.keys())

    features = []
    masks_small = []
    kept_ids = []
    for img_id in img_ids:
        info = coco.loadImgs(img_id)[0]
        gray = cv2.imread(os.path.join(img_dir, info["file_name"]), 0)
        if gray is None:
            continue
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        mask = _mask_from_annotations(anns, info["height"], info["width"])
        features.append(_appearance_feature(gray, feat_size))
        masks_small.append(cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST).astype(np.float32))
        kept_ids.append(img_id)

    features = np.stack(features, axis=0)
    masks_small = np.stack(masks_small, axis=0)

    feat_mean = features.mean(axis=0)
    feat_std = features.std(axis=0) + 1e-6
    feats_std = (features - feat_mean) / feat_std

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(feats_std)

    prototypes = np.zeros((n_clusters, img_size, img_size), dtype=np.float32)
    centroids = np.zeros((n_clusters, features.shape[1]), dtype=np.float32)
    for k in range(n_clusters):
        sel = labels == k
        if sel.sum() == 0:
            continue
        prototypes[k] = masks_small[sel].mean(axis=0)
        centroids[k] = feats_std[sel].mean(axis=0)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, PRIOR_FILE_NAME)
    np.savez(
        out_path,
        prototypes=prototypes,
        centroids=centroids,
        feat_mean=feat_mean.astype(np.float32),
        feat_std=feat_std.astype(np.float32),
        feat_size=np.int64(feat_size),
        img_size=np.int64(img_size),
        temperature=np.float32(temperature),
    )
    counts = [int((labels == k).sum()) for k in range(n_clusters)]
    print(f"Saved tongue prior to: {out_path}")
    print(f"Frames used: {len(kept_ids)}, cluster sizes: {counts}")
    return out_path


class TonguePrior:
    """Runtime helper that turns a frame into 4 similarity-weighted shape prototypes."""

    def __init__(self, prior_path=None):
        if prior_path is None:
            prior_path = os.path.join(DEFAULT_PRIOR_DIR, PRIOR_FILE_NAME)
        data = np.load(prior_path)
        self.prototypes = data["prototypes"].astype(np.float32)      # (K, img_size, img_size)
        self.centroids = data["centroids"].astype(np.float32)        # (K, feat_dim)
        self.feat_mean = data["feat_mean"].astype(np.float32)
        self.feat_std = data["feat_std"].astype(np.float32)
        self.feat_size = int(data["feat_size"])
        self.img_size = int(data["img_size"])
        self.temperature = float(data["temperature"])
        self.n_clusters = self.prototypes.shape[0]

    @staticmethod
    def default_path():
        return os.path.join(DEFAULT_PRIOR_DIR, PRIOR_FILE_NAME)

    @staticmethod
    def exists(prior_path=None):
        if prior_path is None:
            prior_path = TonguePrior.default_path()
        return os.path.exists(prior_path)

    def weights(self, gray_image):
        """Softmax cosine-similarity weights (K,) between the frame and cluster centroids."""
        feat = _appearance_feature(gray_image, self.feat_size)
        feat = (feat - self.feat_mean) / self.feat_std
        fn = feat / (np.linalg.norm(feat) + 1e-8)
        cn = self.centroids / (np.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-8)
        sims = cn @ fn                                  # (K,) cosine similarity in [-1, 1]
        logits = sims / max(self.temperature, 1e-6)
        logits -= logits.max()
        exp = np.exp(logits)
        return (exp / (exp.sum() + 1e-8)).astype(np.float32)

    def weighted_prototypes(self, gray_image, out_hw=None):
        """Return (K, H, W) prototypes scaled by similarity weights, resized to out_hw."""
        w = self.weights(gray_image)
        protos = self.prototypes * w[:, None, None]
        if out_hw is not None and (protos.shape[1], protos.shape[2]) != tuple(out_hw):
            resized = np.stack(
                [cv2.resize(protos[k], (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LINEAR)
                 for k in range(protos.shape[0])],
                axis=0,
            )
            protos = resized
        return protos.astype(np.float32)


def perturb_mask(mask, rng=None, p_apply=0.9, max_shift_frac=0.04, p_dropout=0.1, noise_std=0.05):
    """Perturb a binary/soft mask to mimic prediction error for temporal teacher forcing.

    Applies random morphology (dilate/erode), a small translation, optional full
    dropout (returns zeros), and light gaussian noise. Returns float32 in [0, 1].
    """
    rng = rng or np.random
    mask = mask.astype(np.float32)
    if mask.max() > 1.0:
        mask = mask / 255.0

    # Random full dropout: forces the model not to over-rely on the temporal prior
    # and mimics the inference cold-start / lost-track situation.
    if rng.rand() < p_dropout:
        return np.zeros_like(mask, dtype=np.float32)

    if rng.rand() < p_apply:
        binary = (mask > 0.5).astype(np.uint8)
        ksz = int(rng.choice([3, 5, 7]))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        if rng.rand() < 0.5:
            binary = cv2.dilate(binary, kernel, iterations=1)
        else:
            binary = cv2.erode(binary, kernel, iterations=1)
        mask = binary.astype(np.float32)

        h, w = mask.shape[:2]
        max_shift = max(1, int(max_shift_frac * max(h, w)))
        tx = int(rng.randint(-max_shift, max_shift + 1))
        ty = int(rng.randint(-max_shift, max_shift + 1))
        m = np.float32([[1, 0, tx], [0, 1, ty]])
        mask = cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

    if noise_std > 0:
        mask = mask + rng.normal(0.0, noise_std, size=mask.shape).astype(np.float32)
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def _parse_args():
    parser = argparse.ArgumentParser(description="Build tongue shape prototypes + appearance centroids.")
    parser.add_argument("--data_path", type=str, required=True, help="Repo root containing dataset.json and image/.")
    parser.add_argument("--out_dir", type=str, default=DEFAULT_PRIOR_DIR)
    parser.add_argument("--n_clusters", type=int, default=N_CLUSTERS)
    parser.add_argument("--feat_size", type=int, default=FEAT_SIZE)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_priors(
        data_path=args.data_path,
        out_dir=args.out_dir,
        n_clusters=args.n_clusters,
        feat_size=args.feat_size,
        img_size=args.img_size,
        temperature=args.temperature,
        seed=args.seed,
    )
