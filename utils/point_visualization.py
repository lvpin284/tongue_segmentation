import pandas as pd
import cv2
import numpy as np
import glob
import os

def create_videos_from_csv(results_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(results_dir, '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return

    print(f"Found {len(csv_files)} CSV files. Starting processing...")

    for csv_file in csv_files:
        print(f"Processing {os.path.basename(csv_file)}...")
        try:
            df = pd.read_csv(csv_file)
            
            # Identify coordinate columns
            x_cols = [c for c in df.columns if c.startswith('x') and c[1:].isdigit()]
            y_cols = [c for c in df.columns if c.startswith('y') and c[1:].isdigit()]
            
            if not x_cols or not y_cols:
                print(f"Skipping {csv_file}: No coordinate columns found.")
                continue

            # Determine video resolution based on max coordinates
            # Add some padding
            max_x = df[x_cols].max().max()
            max_y = df[y_cols].max().max()
            
            if pd.isna(max_x) or pd.isna(max_y):
                print(f"Skipping {csv_file}: No valid coordinates found.")
                continue

            width = int(max_x) + 50
            height = int(max_y) + 50
            
            # Ensure even dimensions for video encoding
            width = width if width % 2 == 0 else width + 1
            height = height if height % 2 == 0 else height + 1
            
            # Use fixed resolution if preferred, e.g., 1280x720, but dynamic is safer for varying inputs
            # width, height = 1280, 720 

            fps = 25  # standard frame rate
            video_name = os.path.basename(csv_file).replace('.csv', '.mp4')
            video_path = os.path.join(output_dir, video_name)
            
            # Initialize VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            
            if not video.isOpened():
                print(f"Error: Could not create video writer for {video_path}")
                continue

            num_points = min(len(x_cols), len(y_cols))

            for index, row in df.iterrows():
                # Create a white background image
                img = np.ones((height, width, 3), dtype=np.uint8) * 255
                
                # Draw points
                for i in range(num_points):
                    x_col = f'x{i}'
                    y_col = f'y{i}'
                    
                    if x_col in df.columns and y_col in df.columns:
                        x = row[x_col]
                        y = row[y_col]
                        
                        if not pd.isna(x) and not pd.isna(y):
                            # Draw a red circle
                            cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)
                
                # Add frame index text (optional)
                frame_idx = row.get('frame_idx', index)
                cv2.putText(img, f'Frame: {frame_idx}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                
                video.write(img)
            
            video.release()
            print(f"Successfully saved video to {video_path}")
            
        except Exception as e:
            print(f"Failed to process {csv_file}: {e}")

if __name__ == "__main__":
    # Adjust paths based on the user's workspace structure
    base_dir = '/data/projects/tongue_segmentation/code_represent'
    results_directory = os.path.join(base_dir, 'testvedio/results')
    output_directory = os.path.join(base_dir, 'testvedio/results_videos')
    
    create_videos_from_csv(results_directory, output_directory)
