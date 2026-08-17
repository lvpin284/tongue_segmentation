import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import os
import numpy as np

def create_3d_plots(results_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(results_dir, '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return

    print(f"Found {len(csv_files)} CSV files. Starting 3D plotting...")

    for csv_file in csv_files:
        try:
            filename = os.path.basename(csv_file)
            print(f"Processing {filename}...")
            
            df = pd.read_csv(csv_file)
            
            # Identify coordinate columns
            x_cols = [c for c in df.columns if c.startswith('x') and c[1:].isdigit()]
            y_cols = [c for c in df.columns if c.startswith('y') and c[1:].isdigit()]
            
            if not x_cols or not y_cols:
                print(f"Skipping {csv_file}: No coordinate columns found.")
                continue

            num_points = min(len(x_cols), len(y_cols))
            
            # Prepare data for plotting
            all_x = []
            all_y = []
            all_t = []
            
            # We want to plot (x, y, t)
            # t is the index (or frame_idx column if it exists)
            
            for index, row in df.iterrows():
                t = row.get('frame_idx', index)
                
                for i in range(num_points):
                    x_col = f'x{i}'
                    y_col = f'y{i}'
                    
                    if x_col in df.columns and y_col in df.columns:
                        x = row[x_col]
                        y = row[y_col]
                        
                        if not pd.isna(x) and not pd.isna(y):
                            all_x.append(x)
                            all_y.append(y)
                            all_t.append(t)
            
            if not all_x:
                print(f"Skipping {csv_file}: No valid points to plot.")
                continue

            # Create 3D plot
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Scatter plot
            # c=all_t helps execute color gradient by time
            scatter = ax.scatter(all_x, all_y, all_t, c=all_t, cmap='viridis', s=2, alpha=0.6)
            
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_zlabel('Time (Frame Index)')
            ax.set_title(f'3D Trajectory (X, Y, Time) - {filename}')
            
            # Start view from an angle that shows the time progression clearly
            ax.view_init(elev=20., azim=-45)

            # Invert Y axis because image coordinates (0,0) are top-left usually, 
            # while plots are bottom-left. This aligns visualization with image view roughly.
            ax.invert_yaxis()

            # Add a color bar which maps values to colors
            cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.7)
            cbar.set_label('Time (Frame)')
            
            output_path_png = os.path.join(output_dir, filename.replace('.csv', '_3d.png'))
            plt.savefig(output_path_png, dpi=150)
            plt.close()
            
            print(f"Saved 3D plot to {output_path_png}")

        except Exception as e:
            print(f"Failed to process {csv_file}: {e}")

if __name__ == "__main__":
    # Adjust paths based on the user's workspace structure
    base_dir = '/data/projects/tongue_segmentation/code_represent'
    results_directory = os.path.join(base_dir, 'testvedio/results')
    output_directory = os.path.join(base_dir, 'testvedio/results_3d_plots')
    
    create_3d_plots(results_directory, output_directory)
