import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

def read_metrics(file, model_name):
    df = pd.read_csv(file)
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    if model_name == 'YOLOv9':
        # Map YOLO columns to standard names
        # YOLO columns: epoch, train/box_loss, train/seg_loss, ..., metrics/mAP50(M), metrics/mAP50-95(M), ...
        # We use val/seg_loss for loss, mAP50(M) for dice (proxy), mAP50-95(M) for iou (proxy)
        
        # Check available columns
        # print(df.columns)
        
        new_df = pd.DataFrame()
        new_df['epoch'] = df['epoch']
        
        if 'val/seg_loss' in df.columns:
            new_df['loss'] = df['val/seg_loss']
        elif 'train/seg_loss' in df.columns:
            new_df['loss'] = df['train/seg_loss']
        else:
            new_df['loss'] = 0
            
        if 'metrics/mAP50(M)' in df.columns:
            new_df['dice'] = df['metrics/mAP50(M)'] # Proxy
        else:
            new_df['dice'] = 0
            
        if 'metrics/mAP50-95(M)' in df.columns:
            new_df['iou'] = df['metrics/mAP50-95(M)'] # Proxy
        else:
            new_df['iou'] = 0
            
        return new_df
    else:
        return df

def plot_metrics(metrics_files, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 1. Learning Curves (Loss)
    plt.figure(figsize=(10, 6))
    for name, file in metrics_files.items():
        if os.path.exists(file):
            df = read_metrics(file, name)
            plt.plot(df['epoch'], df['loss'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_comparison.png'))
    plt.close()

    # 2. Dice Score Comparison
    plt.figure(figsize=(10, 6))
    for name, file in metrics_files.items():
        if os.path.exists(file):
            df = read_metrics(file, name)
            plt.plot(df['epoch'], df['dice'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score (mAP50 for YOLO)')
    plt.title('Validation Dice Score Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'dice_comparison.png'))
    plt.close()
    
    # 3. IoU Score Comparison
    plt.figure(figsize=(10, 6))
    for name, file in metrics_files.items():
        if os.path.exists(file):
            df = read_metrics(file, name)
            plt.plot(df['epoch'], df['iou'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('IoU Score (mAP50-95 for YOLO)')
    plt.title('Validation IoU Score Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'iou_comparison.png'))
    plt.close()

    # 4. Bar Chart for Final Metrics
    final_metrics = {'Model': [], 'Dice': [], 'IoU': []}
    for name, file in metrics_files.items():
        if os.path.exists(file):
            df = read_metrics(file, name)
            final_metrics['Model'].append(name)
            final_metrics['Dice'].append(df['dice'].iloc[-1])
            final_metrics['IoU'].append(df['iou'].iloc[-1])
    
    if final_metrics['Model']:
        df_final = pd.DataFrame(final_metrics)
        df_final.set_index('Model', inplace=True)
        
        ax = df_final.plot(kind='bar', figsize=(10, 6), rot=0)
        plt.title('Final Model Comparison')
        plt.ylabel('Score')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'final_comparison_bar.png'))
        plt.close()

def plot_ablation(file_with_tv, file_without_tv, output_dir):
    print(f"Running plot_ablation with {file_with_tv} and {file_without_tv}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if not os.path.exists(file_with_tv) or not os.path.exists(file_without_tv):
        print("Ablation files missing, skipping ablation plot.")
        return

    df_tv = pd.read_csv(file_with_tv)
    df_no_tv = pd.read_csv(file_without_tv)
    print(f"Loaded dataframes. TV: {len(df_tv)}, No TV: {len(df_no_tv)}")
    
    # Plot IoU Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(df_tv['step'], df_tv['iou'], label='With TV Loss (Ours)', color='green')
    plt.plot(df_no_tv['step'], df_no_tv['iou'], label='Without TV Loss', color='orange', linestyle='--')
    plt.xlabel('Step')
    plt.ylabel('IoU Score')
    plt.title('Ablation Study: Impact of TV Loss on IoU')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'ablation_iou.png'))
    plt.close()
    
    # Plot Loss Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(df_tv['step'], df_tv['loss'], label='With TV Loss (Ours)', color='green')
    plt.plot(df_no_tv['step'], df_no_tv['loss'], label='Without TV Loss', color='orange', linestyle='--')
    plt.xlabel('Step')
    plt.ylabel('Total Loss')
    plt.title('Ablation Study: Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'ablation_loss.png'))
    plt.close()

def generate_dummy_data():
    # Generate dummy data if files don't exist for visualization
    epochs = 20
    
    # UNet
    unet_loss = np.linspace(0.8, 0.2, epochs) + np.random.normal(0, 0.02, epochs)
    unet_dice = np.linspace(0.5, 0.85, epochs) + np.random.normal(0, 0.01, epochs)
    unet_iou = np.linspace(0.4, 0.75, epochs) + np.random.normal(0, 0.01, epochs)
    pd.DataFrame({'epoch': range(1, epochs+1), 'loss': unet_loss, 'dice': unet_dice, 'iou': unet_iou}).to_csv('unet_metrics.csv', index=False)
    
    # CNN
    cnn_loss = np.linspace(0.9, 0.4, epochs) + np.random.normal(0, 0.03, epochs)
    cnn_dice = np.linspace(0.4, 0.70, epochs) + np.random.normal(0, 0.02, epochs)
    cnn_iou = np.linspace(0.3, 0.60, epochs) + np.random.normal(0, 0.02, epochs)
    pd.DataFrame({'epoch': range(1, epochs+1), 'loss': cnn_loss, 'dice': cnn_dice, 'iou': cnn_iou}).to_csv('cnn_metrics.csv', index=False)
    
    # MedSAM
    sam_loss = np.linspace(0.7, 0.15, epochs) + np.random.normal(0, 0.01, epochs)
    sam_dice = np.linspace(0.6, 0.92, epochs) + np.random.normal(0, 0.005, epochs)
    sam_iou = np.linspace(0.5, 0.85, epochs) + np.random.normal(0, 0.005, epochs)
    pd.DataFrame({'epoch': range(1, epochs+1), 'loss': sam_loss, 'dice': sam_dice, 'iou': sam_iou}).to_csv('medsam_metrics.csv', index=False)
    
    # YOLO (Simulated)
    yolo_loss = np.linspace(0.6, 0.18, epochs) + np.random.normal(0, 0.02, epochs)
    yolo_dice = np.linspace(0.55, 0.88, epochs) + np.random.normal(0, 0.01, epochs)
    yolo_iou = np.linspace(0.45, 0.80, epochs) + np.random.normal(0, 0.01, epochs)
    pd.DataFrame({'epoch': range(1, epochs+1), 'loss': yolo_loss, 'dice': yolo_dice, 'iou': yolo_iou}).to_csv('yolo_metrics.csv', index=False)

if __name__ == "__main__":
    print("Starting plot_experiment.py...")
    # Check if we have data, otherwise generate dummy data
    files = {
        'UNet': 'unet_metrics.csv',
        'SimpleCNN': 'cnn_metrics.csv',
        'MedSAM': 'medsam_metrics.csv',
        'YOLOv9': 'yolo_metrics.csv',
        'SAM2 (Ours)': '/data/projects/tongue_segmentation/sam2/sam2_metrics.csv'
    }
    
    # Only generate dummy data if NO files exist at all
    existing_files = [f for f in files.values() if os.path.exists(f)]
    if not existing_files:
        print("No log files found. Generating dummy data for demonstration.")
        generate_dummy_data()
        
    output_dir = os.path.abspath('plots')
    print(f"Saving plots to: {output_dir}")
    
    plot_metrics(files, output_dir)
    
    # Run Ablation Plotting
    plot_ablation(
        '/data/projects/tongue_segmentation/sam2/sam2_metrics.csv',
        '/data/projects/tongue_segmentation/sam2/sam2_no_tv_metrics.csv',
        output_dir
    )
    
    print("Plots saved to 'plots' directory.")
