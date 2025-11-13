import zarr
import numpy as np
import plotly.graph_objects as go
import os
import sys
from math import ceil

# --- CONFIGURATION ---
ZARR_FILE = 'neuron_labels.zarr'
HTML_FILE = 'neuron_volume_3d.html'

# Downsampling factor. 8 = read every 8th pixel in each direction.
DOWNSAMPLE_FACTOR = 8

# Voxel size (nm) from inspect_ims.py (Z, Y, X)
VOXEL_SIZE_ZYX = np.array([6.94444444, 3.90625, 3.90625])

# --- CROP CONFIGURATION ---
# Set to False to scan the whole volume
CROP_CENTER = False
CROP_SHAPE = (10, 64, 64) # (Z, Y, X)
# ---------------------

def calculate_aspect_ratio(shape, voxel_size):
    """Calculates the physical aspect ratio for the plot."""
    physical_size = np.array(shape) * voxel_size
    if physical_size.size == 0 or physical_size[2] == 0: 
        return dict(x=1, y=1, z=1)
    aspect_ratio = physical_size / physical_size[2]
    return dict(x=aspect_ratio[2], y=aspect_ratio[1], z=aspect_ratio[0])
    

def visualize_zarr_volume_maxpool(zarr_path, html_path):
    print(f"--- Starting Volume Visualization: {zarr_path} ---")
    
    if not os.path.exists(zarr_path):
        print(f"FATAL ERROR: Zarr file not found at: {zarr_path}")
        print("Please run 'process_filaments.py' first to create it.")
        sys.exit(1)

    # Open the Zarr array
    za = zarr.open(zarr_path, mode='r')
    print(f"  Opened Zarr. Full shape: {za.shape}, Dtype: {za.dtype}")

    # --- Downsampling with Max Pooling ---
    print(f"  Downsampling by {DOWNSAMPLE_FACTOR}x using Max-Pooling...")
    
    full_downsampled_shape = tuple(
        ceil(s / DOWNSAMPLE_FACTOR) for s in za.shape
    )
    
    if CROP_CENTER:
        print(f"  Calculating crop for central {CROP_SHAPE} region...")
        center_z, center_y, center_x = [s // 2 for s in full_downsampled_shape]
        half_z, half_y, half_x = [s // 2 for s in CROP_SHAPE]
        z_start, z_end = max(0, center_z - half_z), min(full_downsampled_shape[0], center_z + half_z)
        y_start, y_end = max(0, center_y - half_y), min(full_downsampled_shape[1], center_y + half_y)
        x_start, x_end = max(0, center_x - half_x), min(full_downsampled_shape[2], center_x + half_x)
        volume_shape = (z_end - z_start, y_end - y_start, x_end - x_start)
        print(f"  Final cropped shape: {volume_shape}")
        plot_offset = [z_start, y_start, x_start]
    else:
        z_start, z_end = 0, full_downsampled_shape[0]
        y_start, y_end = 0, full_downsampled_shape[1]
        x_start, x_end = 0, full_downsampled_shape[2]
        volume_shape = full_downsampled_shape
        plot_offset = [0, 0, 0]
        print(f"  Processing full downsampled shape: {volume_shape}")

    volume = np.zeros(volume_shape, dtype=za.dtype)
    df = DOWNSAMPLE_FACTOR
    
    for z_idx, z in enumerate(range(z_start, z_end)):
        print(f"    - Processing Z-slice {z_idx+1}/{volume_shape[0]} (Global Z: {z})...", end="\r")
            
        for y_idx, y in enumerate(range(y_start, y_end)):
            for x_idx, x in enumerate(range(x_start, x_end)):
                z_full_start, z_full_end = z * df, min((z + 1) * df, za.shape[0])
                y_full_start, y_full_end = y * df, min((y + 1) * df, za.shape[1])
                x_full_start, x_full_end = x * df, min((x + 1) * df, za.shape[2])
                try:
                    block = za[z_full_start:z_full_end, y_full_start:y_full_end, x_full_start:x_full_end]
                    if block.size > 0:
                        volume[z_idx, y_idx, x_idx] = np.max(block)
                except Exception as e:
                    print(f"  Error reading block at {(z,y,x)}: {e}")
                    pass 
        
    print("\n  Downsampling complete.")

    # Find all non-zero pixels
    non_zero_indices = np.argwhere(volume > 0)
    
    if non_zero_indices.shape[0] == 0:
        print("  No labels found in this region. The Zarr file might be empty or the crop is too small.")
        return
    
    # Get the coordinates (Z, Y, X)
    zz = non_zero_indices[:, 0]
    yy = non_zero_indices[:, 1]
    xx = non_zero_indices[:, 2]
    
    # Get the label ID for each point
    labels = volume[zz, yy, xx]
    
    # If we cropped, add the offset back
    if CROP_CENTER:
        zz = zz + z_start
        yy = yy + y_start
        xx = xx + x_start
        
    print(f"  Found {len(xx)} downsampled points to plot.")

    # --- Create 3D Scatter Plot ---
    fig = go.Figure()
    
    aspect_ratio = calculate_aspect_ratio(volume_shape, VOXEL_SIZE_ZYX * DOWNSAMPLE_FACTOR)
    print(f"  Setting aspect ratio: {aspect_ratio}")

    fig.add_trace(go.Scatter3d(
        x=xx, # X
        y=yy, # Y
        z=zz, # Z
        mode='markers',
        marker=dict(
            # --- THIS IS THE FIX ---
            # Make the marker size equal to the downsample factor
            # so they appear to "touch"
            size=DOWNSAMPLE_FACTOR,
            # --- END FIX ---
            color=labels, # Color each point by its label ID
            colorscale='Viridis',
            opacity=1.0 # Make them solid
        ),
        name='Neurons'
    ))

    print("  Finalizing 3D plot...")
    
    plot_title = "3D Point Cloud of Rasterized Neurons (Max-Pooled)"
    if CROP_CENTER:
        plot_title += " (Central Crop)"

    fig.update_layout(
        title=plot_title,
        scene=dict(
            xaxis_title="X (Downsampled Voxel)",
            yaxis_title="Y (Downsampled Voxel)",
            zaxis_title="Z (Downsampled Voxel)",
            aspectmode='manual', # Use our calculated aspect ratio
            aspectratio=aspect_ratio
        )
    )

    fig.write_html(html_path)
    print(f"\n✅ Successfully saved 3D point cloud plot to: {html_path}")
    print("   You can now open this file in your web browser.")


if __name__ == "__main__":
    try:
        visualize_zarr_volume_maxpool(ZARR_FILE, HTML_FILE)
    except Exception as e:
        print(f"\n--- A FATAL ERROR OCCURRED ---")
        print(f"{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)