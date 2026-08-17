import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ["MPLBACKEND"] = "Agg"
from pathlib import Path
from types import SimpleNamespace
import cv2, matplotlib.pyplot as plt, numpy as np
import torch, torch.nn.functional as F

ROOT = Path(__file__).parent
IMG_PATH = ROOT / "frame_000610.jpg"
SAM_CKPT = ROOT / "checkpoints/sam_vit_b_01ec64.pth"
CKPT_DIR = (ROOT / "../save/Tongue").resolve()
IMG_SIZE = 256
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not IMG_PATH.exists():
    for p in sorted(ROOT.glob("*.jpg")) + sorted(ROOT.glob("*.png")):
        IMG_PATH = p; break
print(f"Image: {IMG_PATH}", flush=True)

sys.path.insert(0, str(ROOT))
from models.model_dict import get_model
from utils.config import get_config

ckpt_candidates = sorted(CKPT_DIR.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
CKPT_PATH = ckpt_candidates[0]
print(f"Checkpoint: {CKPT_PATH.name}", flush=True)

args = SimpleNamespace(
    modelname="TongueSegSAM", encoder_input_size=IMG_SIZE, low_image_size=128,
    task="Cardiac_multi_plane_test", vit_name="vit_b", sam_ckpt=str(SAM_CKPT),
    batch_size=1, n_gpu=1, base_lr=1e-4)
opt = get_config(args.task); opt.device = DEVICE
model = get_model(args.modelname, args=args, opt=opt).to(DEVICE)
state = torch.load(str(CKPT_PATH), map_location=DEVICE)
state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
model.load_state_dict(state, strict=False); model.eval()
print("Model loaded.", flush=True)


def build_prototype_prior(img_gray):
    img_f = img_gray.astype(np.float32) / 255.0
    ch1 = img_f
    ch2 = cv2.GaussianBlur(img_f, (0, 0), 2.0)
    sx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    ch3 = cv2.normalize(np.abs(sx) + np.abs(sy), None, 0.0, 1.0, cv2.NORM_MINMAX)
    _, th = cv2.threshold((img_f * 255).astype(np.uint8), 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ch4 = th.astype(np.float32) / 255.0
    return np.stack([ch1, ch2, ch3, ch4], axis=0)


img0 = cv2.imread(str(IMG_PATH), cv2.IMREAD_GRAYSCALE)
img  = cv2.resize(img0, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
img_tensor = torch.from_numpy(img).float().div(255.0).unsqueeze(0).unsqueeze(0).to(DEVICE)
pt = (torch.tensor([[[IMG_SIZE // 2, IMG_SIZE // 2]]], dtype=torch.float32).to(DEVICE),
      torch.tensor([[1]], dtype=torch.int64).to(DEVICE))
cls_sim_avg_np = build_prototype_prior(img)
cls_sim_avg = torch.from_numpy(cls_sim_avg_np).unsqueeze(0).to(dtype=torch.float32, device=DEVICE)

with torch.no_grad():
    image_embeddings, image_cnn_features = model.image_encoder(img_tensor)
    sparse_emb, dense_emb, low_coarse_mask_logit, _, _ = model.prompt_encoder(
        points=pt, boxes=None, masks=None, cls_sim_avg_label_input=cls_sim_avg)
    low_res_logits, _ = model.mask_decoder(
        image_embeddings=image_embeddings,
        image_cnn_features=image_cnn_features,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=False)
    final_logits = F.interpolate(low_res_logits, (IMG_SIZE, IMG_SIZE),
                                 mode="bilinear", align_corners=False)

coarse_prob_raw = torch.sigmoid(low_coarse_mask_logit)[0, 0].detach().cpu().numpy()
coarse_prob = cv2.resize(coarse_prob_raw, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
final_prob  = torch.sigmoid(final_logits)[0, 0].detach().cpu().numpy()
print(f"coarse_prob range=[{coarse_prob.min():.3f},{coarse_prob.max():.3f}]", flush=True)
print(f"final_prob  range=[{final_prob.min():.3f},{final_prob.max():.3f}]", flush=True)

threshold  = 0.5
coarse_bin = (coarse_prob > threshold).astype(np.uint8)
final_bin  = (final_prob  > threshold).astype(np.uint8)
cm = cls_sim_avg_np.mean(axis=0)
cm = (cm - cm.min()) / (cm.max() - cm.min() + 1e-8)
proto_maps  = cls_sim_avg_np.copy()
proto_names = ["Raw Prior", "Smooth Prior", "Edge Prior", "Otsu Prior"]

# ── Figure 1: main pipeline ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(img, cmap="gray")
axes[0].set_title("Input Image", fontsize=13); axes[0].axis("off")
im1 = axes[1].imshow(cm, cmap="hot")
axes[1].set_title("Center Mask\n(Weighted Prototype Mean)", fontsize=13)
axes[1].axis("off"); plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
im2 = axes[2].imshow(coarse_prob, cmap="viridis")
axes[2].set_title(f"Coarse Prediction\n(Prob, t={threshold:.2f})", fontsize=13)
axes[2].axis("off"); plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
im3 = axes[3].imshow(final_prob, cmap="magma")
axes[3].set_title(f"Final Prediction\n(Prob, t={threshold:.2f})", fontsize=13)
axes[3].axis("off"); plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
plt.suptitle("TongueSegSAM — Intermediate Visualization", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(str(ROOT / "mask_fig1_pipeline.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved mask_fig1_pipeline.png", flush=True)

# ── Figure 2: binary overlays ────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
c1 = np.stack([img, img, img], axis=-1).astype(np.uint8); c1[coarse_bin > 0] = [0, 200, 0]
axes2[0].imshow(c1); axes2[0].set_title("Coarse Overlay (green)", fontsize=13); axes2[0].axis("off")
c2 = np.stack([img, img, img], axis=-1).astype(np.uint8); c2[final_bin > 0] = [255, 80, 0]
axes2[1].imshow(c2); axes2[1].set_title("Final Overlay (orange)", fontsize=13); axes2[1].axis("off")
c3 = np.stack([img, img, img], axis=-1).astype(np.uint8)
c3[coarse_bin > 0] = [0, 200, 0]
c3[final_bin  > 0] = [255, 80, 0]
c3[(coarse_bin > 0) & (final_bin > 0)] = [0, 180, 255]
axes2[2].imshow(c3)
axes2[2].set_title("Overlap\n(green=coarse, orange=final, cyan=both)", fontsize=11)
axes2[2].axis("off")
plt.tight_layout()
plt.savefig(str(ROOT / "mask_fig2_overlay.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved mask_fig2_overlay.png", flush=True)

# ── Figure 3: prototype channels ─────────────────────────────────────────────
fig3, axes3 = plt.subplots(1, 4, figsize=(18, 4))
for i, name in enumerate(proto_names):
    p = proto_maps[i].copy(); p = (p - p.min()) / (p.max() - p.min() + 1e-8)
    im = axes3[i].imshow(p, cmap="inferno")
    axes3[i].set_title(f"Prototype {i+1}\n({name})", fontsize=11)
    axes3[i].axis("off"); plt.colorbar(im, ax=axes3[i], fraction=0.046, pad=0.04)
plt.suptitle("Prototypes (cls_sim_avg_label_input, 4 channels)", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(str(ROOT / "mask_fig3_prototypes.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved mask_fig3_prototypes.png", flush=True)
print("ALL DONE.", flush=True)
