import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from dataset import TongueDataset
from models.simple_cnn import SimpleCNN
from utils import dice_coeff, iou_score, MetricLogger

# Configuration
DATA_DIR = "/data/projects/tongue_segmentation/sam2/dataset"
IMAGES_DIR = os.path.join(DATA_DIR, "image")
MASKS_FILE = os.path.join(DATA_DIR, "dataset.json")
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = int(os.environ.get("MAX_EPOCHS", 20))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    total_loss = 0
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE).float()
        targets = targets.float().unsqueeze(1).to(DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        total_loss += loss.item()
    
    return total_loss / len(loader)

def check_accuracy(loader, model, device="cuda"):
    dice_score = 0
    iou = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            dice_score += dice_coeff(preds, y)
            iou += iou_score(preds, y)

    avg_dice = dice_score/len(loader)
    avg_iou = iou/len(loader)
    print(f"Dice Score: {avg_dice}")
    print(f"IoU Score: {avg_iou}")
    model.train()
    return avg_dice, avg_iou

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    # Re-doing dataset split properly
    full_dataset = TongueDataset(root_dir=IMAGES_DIR, annotation_file=MASKS_FILE, transform=None)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])
    
    train_dataset = TongueDataset(root_dir=IMAGES_DIR, annotation_file=MASKS_FILE, transform=train_transform)
    train_dataset.ids = [full_dataset.ids[i] for i in train_subset.indices]
    
    val_dataset = TongueDataset(root_dir=IMAGES_DIR, annotation_file=MASKS_FILE, transform=val_transform)
    val_dataset.ids = [full_dataset.ids[i] for i in val_subset.indices]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = SimpleCNN(n_channels=3, n_classes=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    
    logger = MetricLogger("cnn_metrics.csv", ["epoch", "loss", "dice", "iou"])

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        avg_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        avg_dice, avg_iou = check_accuracy(val_loader, model, device=DEVICE)
        
        logger.log({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "dice": avg_dice.item() if torch.is_tensor(avg_dice) else avg_dice,
            "iou": avg_iou.item() if torch.is_tensor(avg_iou) else avg_iou
        })
        
        # Save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, "cnn_checkpoint.pth.tar")

if __name__ == "__main__":
    main()
