import os
import re
import cv2
import numpy as np
import torch
import json
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from .data_us import correct_dims, random_click, random_bbox, fixed_click, fixed_bbox, to_long_tensor
from .tongue_prior import TonguePrior, perturb_mask, _mask_from_annotations, N_CLUSTERS
from torchvision import transforms as T

_FRAME_NUM_RE = re.compile(r"(\d+)")


def _frame_number(file_name):
    """Parse the frame index encoded in a file name like 'frame_000123.jpg'."""
    matches = _FRAME_NUM_RE.findall(os.path.basename(file_name))
    return int(matches[-1]) if matches else -1


class TongueDataset(Dataset):
    def __init__(self, data_path, split="train", joint_transform=None, img_size=256, prompt="click",
                 use_prior=True, prior_path=None, temporal_prior=True, **kwargs):
        self.data_path = data_path
        self.img_dir = os.path.join(data_path, "image")
        self.json_path = os.path.join(data_path, "dataset.json")
        self.coco = COCO(self.json_path)
        self.split = split
        self.img_size = img_size
        self.prompt = prompt
        self.is_train = 'train' in split

        self.img_ids = list(self.coco.imgs.keys())

        # Build the global temporal order (by frame number) over ALL labeled frames so
        # each frame can look up the previous frame's mask regardless of train/val split.
        ordered = sorted(self.img_ids, key=lambda i: _frame_number(self.coco.loadImgs(i)[0]['file_name']))
        self.prev_id = {}
        for pos, img_id in enumerate(ordered):
            self.prev_id[img_id] = ordered[pos - 1] if pos > 0 else None

        # Simple split: 80% train, 20% val
        np.random.seed(42)
        np.random.shuffle(self.img_ids)
        split_idx = int(0.8 * len(self.img_ids))
        if split == 'train':
            self.ids = self.img_ids[:split_idx]
        else:
            self.ids = self.img_ids[split_idx:]

        # Prior (方案一): 4 shape prototypes + 1 temporal previous-frame mask channel.
        self.temporal_prior = temporal_prior
        self.n_proto = N_CLUSTERS
        self.prior = None
        if use_prior and TonguePrior.exists(prior_path):
            self.prior = TonguePrior(prior_path)
            self.n_proto = self.prior.n_clusters
        elif use_prior:
            print("[TongueDataset] Prior file not found; prototype channels default to zeros. "
                  "Run `python -m utils.tongue_prior --data_path <repo_root>` to build them.")

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y, prior=None: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(self.ids)

    def _load_mask(self, img_id, height, width):
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        return _mask_from_annotations(anns, height, width)

    def _build_prior_stack(self, gray_image, img_id):
        """Return an (H, W, 5) prior: 4 similarity-weighted prototypes + 1 temporal mask."""
        h, w = gray_image.shape[:2]

        if self.prior is not None:
            protos = self.prior.weighted_prototypes(gray_image, out_hw=(h, w))  # (K, H, W)
            proto_hw_c = np.transpose(protos, (1, 2, 0)).astype(np.float32)      # (H, W, K)
        else:
            proto_hw_c = np.zeros((h, w, self.n_proto), dtype=np.float32)

        # Temporal channel: previous labeled frame's GT mask (teacher forcing).
        prev_channel = np.zeros((h, w), dtype=np.float32)
        if self.temporal_prior:
            prev = self.prev_id.get(img_id)
            if prev is not None:
                prev_info = self.coco.loadImgs(prev)[0]
                prev_mask = self._load_mask(prev, prev_info['height'], prev_info['width']).astype(np.float32)
                if prev_mask.shape[:2] != (h, w):
                    prev_mask = cv2.resize(prev_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                if self.is_train:
                    prev_mask = perturb_mask(prev_mask)
                prev_channel = prev_mask.astype(np.float32)

        return np.concatenate([proto_hw_c, prev_channel[:, :, None]], axis=2).astype(np.float32)

    def __getitem__(self, i):
        img_id = self.ids[i]
        img_info = self.coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']

        image = cv2.imread(os.path.join(self.img_dir, file_name), 0)  # Grayscale read
        if image is None:
            # Fallback handling
            image = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        mask = self._load_mask(img_id, image.shape[0], image.shape[1])

        # Build the 5-channel prior in the native image coordinate so it can be
        # geometrically augmented jointly with the image/mask.
        prior_stack = self._build_prior_stack(image, img_id)

        image, mask = correct_dims(image, mask)

        # Apply transformation (prior shares the same geometric augmentation params).
        image, mask, low_mask, prior_tensor = self.joint_transform(image, mask, prior_stack)

        class_id = 1

        if self.prompt == 'click':
            point_label = 1
            if 'train' in self.split:
                pt, point_label = random_click(np.array(mask), class_id)
                bbox = random_bbox(np.array(mask), class_id, self.img_size)
            else:
                pt, point_label = fixed_click(np.array(mask), class_id)
                bbox = fixed_bbox(np.array(mask), class_id, self.img_size)

            mask[mask != class_id] = 0
            mask[mask == class_id] = 1
            low_mask[low_mask != class_id] = 0
            low_mask[low_mask == class_id] = 1

            point_labels = np.array(point_label)

        low_mask = low_mask.unsqueeze(0)
        mask = mask.unsqueeze(0)

        # 5-channel prior: [proto_0..proto_3, previous_frame_mask]
        cls_sim_avg_label_input = prior_tensor  # tensor (5, img_size, img_size)
        cls_sim = np.zeros((1, self.n_proto), dtype=np.float32)

        return {
            'image': image,
            'label': mask,
            'p_label': point_labels,
            'pt': pt,
            'cls_sim': cls_sim,
            'cls_sim_avg_label_input': cls_sim_avg_label_input,
            'bbox': bbox,
            'low_mask': low_mask,
            'image_name': file_name,
            'class_id': class_id,
        }
