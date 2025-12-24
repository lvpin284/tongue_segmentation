import sys
sys.path.append('/data/python-packages')
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
from dataset import TongueDataset
from segment_anything import sam_model_registry
from torch.nn.functional import threshold, normalize
import cv2
from utils import dice_coeff, iou_score, MetricLogger

# Configuration
DATA_DIR = "/data/projects/tongue_segmentation/sam2/dataset"
IMAGES_DIR = os.path.join(DATA_DIR, "image")
MASKS_FILE = os.path.join(DATA_DIR, "dataset.json")
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = int(os.environ.get("MAX_EPOCHS", 20))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_TYPE = "vit_b"
CHECKPOINT_PATH = "/data/projects/tongue_segmentation/ComparativeExperiment/sam_vit_b_01ec64.pth" # User needs to download this

def get_bbox_from_mask(mask):
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0:
        return np.array([0, 0, 1, 1]) # Dummy box
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    
    # Add perturbation
    H, W = mask.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    
    return np.array([x_min, y_min, x_max, y_max])

class MedSAMDataset(TongueDataset):
    def __getitem__(self, index):
        img, mask = super().__getitem__(index)
        # Resize to 1024x1024 for SAM
        img = cv2.resize(img, (1024, 1024))
        mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        
        # Get bbox
        bbox = get_bbox_from_mask(mask)
        
        # Normalize image
        img = (img - np.array([123.675, 116.28, 103.53])) / np.array([58.395, 57.12, 57.375])
        img = torch.tensor(img).permute(2, 0, 1).float()
        
        return img, torch.tensor(mask).float(), torch.tensor(bbox).float()

def main():
    full_dataset = MedSAMDataset(root_dir=IMAGES_DIR, annotation_file=MASKS_FILE, transform=None)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    sam_model = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam_model.to(DEVICE)
    sam_model.train()
    
    # Freeze image encoder
    for param in sam_model.image_encoder.parameters():
        param.requires_grad = False
    for param in sam_model.prompt_encoder.parameters():
        param.requires_grad = False
        
    optimizer = optim.Adam(sam_model.mask_decoder.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss() # Simplified loss
    
    logger = MetricLogger("medsam_metrics.csv", ["epoch", "loss", "dice", "iou"])

    for epoch in range(EPOCHS):
        sam_model.train()
        epoch_loss = 0
        for img, mask, bbox in tqdm(train_loader):
            img = img.to(DEVICE)
            mask = mask.to(DEVICE).unsqueeze(1)
            bbox = bbox.to(DEVICE)
            
            with torch.no_grad():
                image_embedding = sam_model.image_encoder(img)
                
            # Prepare sparse embeddings for bbox
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=bbox.unsqueeze(1),
                masks=None,
            )
            
            # Loop over batch for decoder to handle batching correctly
            low_res_masks_list = []
            iou_predictions_list = []
            
            for i in range(img.shape[0]):
                curr_embedding = image_embedding[i].unsqueeze(0)
                curr_sparse = sparse_embeddings[i].unsqueeze(0)
                curr_dense = dense_embeddings[i].unsqueeze(0)
                
                low_res, iou = sam_model.mask_decoder(
                    image_embeddings=curr_embedding,
                    image_pe=sam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=curr_sparse,
                    dense_prompt_embeddings=curr_dense,
                    multimask_output=False,
                )
                low_res_masks_list.append(low_res)
                iou_predictions_list.append(iou)
            
            low_res_masks = torch.cat(low_res_masks_list, dim=0)
            iou_predictions = torch.cat(iou_predictions_list, dim=0)
            
            # Upscale masks
            upscaled_masks = nn.functional.interpolate(
                low_res_masks,
                size=(1024, 1024),
                mode="bilinear",
                align_corners=False,
            )
            
            loss = loss_fn(upscaled_masks, mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Validation
        sam_model.eval()
        val_dice = 0
        val_iou = 0
        with torch.no_grad():
            for img, mask, bbox in val_loader:
                img = img.to(DEVICE)
                mask = mask.to(DEVICE).unsqueeze(1)
                bbox = bbox.to(DEVICE)
                
                image_embedding = sam_model.image_encoder(img)
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None,
                    boxes=bbox.unsqueeze(1),
                    masks=None,
                )
                
                low_res_masks_list = []
                for i in range(img.shape[0]):
                    curr_embedding = image_embedding[i].unsqueeze(0)
                    curr_sparse = sparse_embeddings[i].unsqueeze(0)
                    curr_dense = dense_embeddings[i].unsqueeze(0)
                    low_res, _ = sam_model.mask_decoder(
                        image_embeddings=curr_embedding,
                        image_pe=sam_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=curr_sparse,
                        dense_prompt_embeddings=curr_dense,
                        multimask_output=False,
                    )
                    low_res_masks_list.append(low_res)
                
                low_res_masks = torch.cat(low_res_masks_list, dim=0)
                upscaled_masks = nn.functional.interpolate(
                    low_res_masks,
                    size=(1024, 1024),
                    mode="bilinear",
                    align_corners=False,
                )
                
                preds = torch.sigmoid(upscaled_masks)
                preds = (preds > 0.5).float()
                val_dice += dice_coeff(preds, mask)
                val_iou += iou_score(preds, mask)
        
        avg_loss = epoch_loss/len(train_loader)
        avg_dice = val_dice/len(val_loader)
        avg_iou = val_iou/len(val_loader)
        
        print(f"Epoch {epoch+1}, Loss: {avg_loss}, Dice: {avg_dice}, IoU: {avg_iou}")
        logger.log({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "dice": avg_dice.item() if torch.is_tensor(avg_dice) else avg_dice,
            "iou": avg_iou.item() if torch.is_tensor(avg_iou) else avg_iou
        })
        
        torch.save(sam_model.state_dict(), "medsam_checkpoint.pth")

if __name__ == "__main__":
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Please download SAM checkpoint {CHECKPOINT_PATH} first!")
    else:
        main()
