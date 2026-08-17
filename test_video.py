import os
import cv2
import torch
import numpy as np
from models.model_dict import get_model
import argparse
from utils.config import get_config
from utils.tongue_prior import TonguePrior


def _load_checkpoint_state_dict(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    new_state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith('module.'):
            new_state_dict[key[7:]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def _resolve_checkpoint_path(explicit_checkpoint, checkpoint_dir):
    if explicit_checkpoint:
        checkpoint_path = os.path.abspath(explicit_checkpoint)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    model_dir = os.path.abspath(checkpoint_dir)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {model_dir}")

    candidates = [os.path.join(model_dir, name) for name in os.listdir(model_dir) if name.endswith('.pth')]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint (*.pth) found under: {model_dir}")

    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', default='TongueSegSAM', type=str)
    parser.add_argument('-encoder_input_size', type=int, default=256)
    parser.add_argument('-low_image_size', type=int, default=128)
    parser.add_argument('--task', default='Cardiac_multi_plane_test')
    parser.add_argument('--vit_name', type=str, default='vit_b')
    parser.add_argument('--sam_ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--base_lr', type=float, default=0.0001)
    parser.add_argument('--checkpoint', type=str, default='', help='Optional explicit model checkpoint path (.pth).')
    parser.add_argument('--checkpoint-dir', type=str, default='../save/Tongue/', help='Directory to auto-pick latest checkpoint when --checkpoint is not provided.')
    parser.add_argument('--video-dir', type=str, default='../testvedio/muler_vedio/vedio2', help='Input video root directory.')
    parser.add_argument('--output-dir', type=str, default='../testvedio/muler_vedio/results', help='Output directory for result videos and points CSV.')
    parser.add_argument('--prior_path', type=str, default=None, help='Path to tongue_prior.npz (defaults to checkpoints/tongue_prior/).')
    
    args = parser.parse_args()
    opt = get_config(args.task)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    opt.device = device
    
    model = get_model(args.modelname, args=args, opt=opt)
    model.to(device)
    
    checkpoint_path = _resolve_checkpoint_path(args.checkpoint, args.checkpoint_dir)
    new_state_dict = _load_checkpoint_state_dict(checkpoint_path, device)
    model.load_state_dict(new_state_dict)
    model.eval()
    print(f"Using segmentation checkpoint: {checkpoint_path}")

    # Load the shape-prototype prior (方案一). Temporal channel is filled autoregressively
    # with the previous frame's predicted mask below.
    tongue_prior = None
    if TonguePrior.exists(args.prior_path):
        tongue_prior = TonguePrior(args.prior_path)
        print(f"Loaded tongue prior with {tongue_prior.n_clusters} prototypes.")
    else:
        print("[test_video] Tongue prior not found; prototype channels default to zeros.")
    n_proto = tongue_prior.n_clusters if tongue_prior is not None else 4

    video_dir = args.video_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Recursively find all mp4 videos in new_video directory
    videos = []
    for root, dirs, files in os.walk(video_dir):
        for f in files:
            if f.endswith('.mp4'):
                videos.append(os.path.join(root, f))
    
    for vid_path in videos:
        vid_name = os.path.basename(vid_path)
        # Create output path maintaining original structure or just unique names
        # Here we just save directly in output_dir, prepended with subfolder to be safe
        rel_path = os.path.relpath(vid_path, video_dir)
        safe_name = rel_path.replace(os.sep, '_')
        
        # Ignore already processed
        if safe_name.startswith('result_'):
            continue
            
        print(f"Processing {vid_path} ...")
        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out_path = os.path.join(output_dir, 'result_' + safe_name)
        csv_path = os.path.join(output_dir, 'points_' + safe_name.replace('.mp4', '.csv'))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        
        frame_idx = 0
        frame_points_records = []
        enc = args.encoder_input_size
        # Autoregressive temporal prior: previous frame's predicted probability mask
        # at encoder resolution. Cold-started with zeros for the first frame.
        prev_prob = np.zeros((enc, enc), dtype=np.float32)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Frame {frame_idx}")

            # Pre-process frame to single-channel tensor input expected by segmentation backend.
            # In test.py: input = torch.randn(1, 1, 256, 256)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
            img_resize = cv2.resize(img, (args.encoder_input_size, args.encoder_input_size))
            img_tensor = torch.from_numpy(img_resize).float() / 255.0
            
            # The network expects [batch, channel, H, W]
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)
            
            # generate prompt: shape [1, 1, 2] # (b, n, 2)
            # central point
            pt_np = np.array([[args.encoder_input_size // 2, args.encoder_input_size // 2]])
            pt_label_np = np.array([1])
            
            pt_coords = torch.as_tensor(pt_np, dtype=torch.float32, device=device).unsqueeze(0)
            pt_labels = torch.as_tensor(pt_label_np, dtype=torch.int, device=device).unsqueeze(0)
            pt = (pt_coords, pt_labels)

            # Build the 5-channel prior: 4 similarity-weighted prototypes + previous-frame mask.
            if tongue_prior is not None:
                protos = tongue_prior.weighted_prototypes(img_resize, out_hw=(enc, enc))  # (K, enc, enc)
            else:
                protos = np.zeros((n_proto, enc, enc), dtype=np.float32)
            prior_np = np.concatenate([protos, prev_prob[None]], axis=0)  # (5, enc, enc)
            cls_sim_avg = torch.from_numpy(prior_np).float().unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(img_tensor, pt, bbox=None, cls_sim_avg_label_input=cls_sim_avg)
                # pred is a dict with 'masks'
                mask_pred = pred['masks']
                # mask_pred shape: [1, 1, 256, 256]; SAM returns logits, so sigmoid -> prob.
                prob_map = torch.sigmoid(mask_pred).squeeze().cpu().numpy()
                mask_np = (prob_map > 0.5).astype(np.uint8)
                # Update temporal prior for the next frame (autoregressive).
                if prob_map.shape != (enc, enc):
                    prev_prob = cv2.resize(prob_map.astype(np.float32), (enc, enc), interpolation=cv2.INTER_LINEAR)
                else:
                    prev_prob = prob_map.astype(np.float32)
                
            mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Overlay green channel
            overlay = frame.copy()
            overlay[mask_resized == 1, 1] = 255 # Set green channel max
            overlay[mask_resized == 1, 0] = 0
            overlay[mask_resized == 1, 2] = 0
            
            # Blend
            final = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            # Extract midpoints and draw continuous curve
            midpoints = []
            for x in range(w):
                ys = np.where(mask_resized[:, x] == 1)[0]
                if len(ys) > 0:
                    mid_y = int(np.mean(ys)) # use mean for more stable center
                    midpoints.append([x, mid_y])
                    
            if len(midpoints) > 10:
                pts = np.array(midpoints, np.int32)
                
                # Fit a polynomial to smooth the curve and remove jaggedness
                x_coords = pts[:, 0]
                y_coords = pts[:, 1]
                
                # A 3rd degree polynomial is robust and perfectly models the tongue's U-shape
                z = np.polyfit(x_coords, y_coords, 3)
                p = np.poly1d(z)
                smooth_y = p(x_coords)
                
                smoothed_pts = np.column_stack((x_coords, smooth_y)).astype(np.int32)
                smoothed_pts = smoothed_pts.reshape((-1, 1, 2))
                
                # Draw the centerline as a connected red curve with thickness 3
                cv2.polylines(final, [smoothed_pts], isClosed=False, color=(0, 0, 255), thickness=3)

                # Extract exactly 100 evenly spaced points along the curve
                min_x = np.min(x_coords)
                max_x = np.max(x_coords)
                xs_100 = np.linspace(min_x, max_x, 100)
                ys_100 = p(xs_100)
                
                # Save the 100 points
                record = [frame_idx]
                for px, py in zip(xs_100, ys_100):
                    record.extend([px, py])
                frame_points_records.append(record)
                
            out.write(final)
            
        cap.release()
        out.release()
        
        # Save points to CSV
        if len(frame_points_records) > 0:
            header = ['frame_idx']
            for i in range(100):
                header.extend([f'x{i}', f'y{i}'])
            
            with open(csv_path, 'w') as f:
                f.write(','.join(header) + '\n')
                for record in frame_points_records:
                    f.write(','.join([f"{val:.3f}" if isinstance(val, float) else str(val) for val in record]) + '\n')

        print(f"Finished {vid_name}")

if __name__ == '__main__':
    main()
