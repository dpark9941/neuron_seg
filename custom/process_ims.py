import h5py
import zarr
import numpy as np
import numcodecs
import sys
import os
import plotly.graph_objects as go
import re # For finding filament groups

# --- SAFETY CAP ---
# Skip drawing any sphere that is larger than 512^3 voxels
# This prevents a single bad vertex radius from crashing the script
MAX_SPHERE_VOXELS = 512 * 512 * 512 # Approx 134 million

def get_voxel_size(f):
    """
    Reads *only* the voxel size.
    We need this to convert the physical Radius to a pixel Radius.
    """
    print("  Reading voxel size metadata...")
    raw_path = "DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data"
    info_path = 'DataSetInfo/Image'
    
    if raw_path not in f:
        raise ValueError(f"Raw data not found at {raw_path}")
    if info_path not in f:
        raise ValueError("'DataSetInfo/Image' group not found.")

    raw_shape_zyx = f[raw_path].shape
    if len(raw_shape_zyx) == 5: # Handle 5D
        raw_shape_zyx = raw_shape_zyx[2:]

    image_info = f[info_path]
    try:
        ext_max_zyx = np.array([
            float(image_info.attrs['ExtMax2'][0]), 
            float(image_info.attrs['ExtMax1'][0]), 
            float(image_info.attrs['ExtMax0'][0])
        ])
        ext_min_zyx = np.array([
            float(image_info.attrs['ExtMin2'][0]), 
            float(image_info.attrs['ExtMin1'][0]), 
            float(image_info.attrs['ExtMin0'][0])
        ])

        physical_size_zyx = ext_max_zyx - ext_min_zyx
        voxel_size_zyx = physical_size_zyx / np.array(raw_shape_zyx)
        
        print(f"    - Voxel Size (μm): {voxel_size_zyx}")
        # Return (Z,Y,X) order
        return voxel_size_zyx

    except Exception as e:
        print(f"  ERROR reading metadata: {e}")
        raise

def draw_sphere_to_zarr(zarr_array, center_zyx_vox, radius_vox_zyx, label_id, vertex_debug_index):
    """
    --- NEW LOW-MEMORY VERSION ---
    Draws a sphere directly into a Zarr array on disk.
    """
    center_z, center_y, center_x = center_zyx_vox[0], center_zyx_vox[1], center_zyx_vox[2]
    radius_z, radius_y, radius_x = radius_vox_zyx
    
    # Create a bounding box around the sphere
    z_min = max(0, int(center_z - radius_z - 1))
    z_max = min(zarr_array.shape[0], int(center_z + radius_z) + 2)
    y_min = max(0, int(center_y - radius_y - 1))
    y_max = min(zarr_array.shape[1], int(center_y + radius_y) + 2)
    x_min = max(0, int(center_x - radius_x - 1))
    x_max = min(zarr_array.shape[2], int(center_x + radius_x) + 2)

    if z_min >= z_max or y_min >= y_max or x_min >= x_max:
        return # Sphere is outside volume
        
    # --- NEW SAFETY CAP ---
    box_size_voxels = (z_max - z_min) * (y_max - y_min) * (x_max - x_min)
    if box_size_voxels > MAX_SPHERE_VOXELS:
        print(f"      - 🔴 SKIPPING Vertex {vertex_debug_index}: Calculated radius is too large!")
        print(f"      -   (Radius (px): z={radius_z:.1f}, y={radius_y:.1f}, x={radius_x:.1f})")
        print(f"      -   (Bounding box: {box_size_voxels} voxels > max {MAX_SPHERE_VOXELS})")
        return
    # --- END SAFETY CAP ---

    # --- ZARR-SPECIFIC LOGIC ---
    try:
        volume_slice = zarr_array[z_min:z_max, y_min:y_max, x_min:x_max]
    except Exception as e:
        print(f"      - ERROR reading zarr slice: {e}")
        return
    # --- END ZARR-SPECIFIC LOGIC ---
    
    # Create coordinate grid for the *in-memory* slice
    zz, yy, xx = np.indices(volume_slice.shape)
    
    # Offset grid to match *global* volume coordinates
    zz += z_min
    yy += y_min
    xx += x_min

    # Calculate distance in *voxels*, scaled by the pixel radius for each axis
    dist_sq = (
        ((zz.astype(np.float32) - center_z) / np.maximum(radius_z, 1e-6))**2 +
        ((yy.astype(np.float32) - center_y) / np.maximum(radius_y, 1e-6))**2 +
        ((xx.astype(np.float32) - center_x) / np.maximum(radius_x, 1e-6))**2
    )
    
    # Create a mask where distance is less than 1
    mask = dist_sq <= 1.0
    
    # Apply the label to the in-memory slice
    # We also only want to draw on background (0)
    final_mask = np.logical_and(mask, volume_slice == 0)
    volume_slice[final_mask] = label_id

    # --- ZARR-SPECIFIC LOGIC ---
    try:
        zarr_array[z_min:z_max, y_min:y_max, x_min:x_max] = volume_slice
    except Exception as e:
        print(f"      - ERROR writing zarr slice: {e}")
    # --- END ZARR-SPECIFIC LOGIC ---


def find_filament_groups(h5_file):
    """
    Recursively finds all groups matching 'FilamentsX'.
    """
    print("  Finding filament groups (recursive search)...")
    filament_groups = []
    
    def visit_func(name, node):
        if isinstance(node, h5py.Group):
            if re.match(r'Filaments\d+', os.path.basename(name)):
                filament_groups.append(node)
                
    h5_file.visititems(visit_func)
    print(f"    - Found {len(filament_groups)} filament groups.")
    return filament_groups


def rasterize_to_zarr(ims_filepath, zarr_filepath):
    """
    --- NEW LOW-MEMORY VERSION ---
    Reads 'neuron.ims', rasterizes all filaments directly to a Zarr file on disk.
    """
    print(f"\n--- 1. Starting Rasterization: {ims_filepath} -> {zarr_filepath} ---")
    
    with h5py.File(ims_filepath, 'r') as f:
        
        # --- Get the geometry of the volume ---
        raw_path = "DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data"
        raw_shape = f[raw_path].shape
        if len(raw_shape) == 5:
            raw_shape = raw_shape[2:] # Get ZYX
        print(f"    - Raw Shape (Z,Y,X): {raw_shape}")
        
        vox_size = get_voxel_size(f)
        
        # --- ZARR-SPECIFIC LOGIC ---
        print(f"  Creating on-disk Zarr file at '{zarr_filepath}'...")
        zarr_chunks = (64, 64, 64) # Good chunking for gunpowder
        label_volume_zarr = zarr.open(
            zarr_filepath,
            mode='w',
            shape=raw_shape,
            dtype=np.uint64,
            chunks=zarr_chunks,
            compressor=numcodecs.GZip(level=5)
        )
        # Removed the .fill(0) line
        
        filament_groups = find_filament_groups(f)
        if not filament_groups:
            print("  No filament groups found. Aborting rasterization.")
            return

        print("\n  --- STARTING RASTERIZATION (Writing to Disk) ---")
        print(f"  Volume Z-shape: 0 to {raw_shape[0]-1}")
        print(f"  Volume Y-shape: 0 to {raw_shape[1]-1}")
        print(f"  Volume X-shape: 0 to {raw_shape[2]-1}")
        
        current_label_id = 1
        total_vertices_drawn = 0
        
        for i, group in enumerate(filament_groups):
            print(f"    - Processing {group.name} as Label ID {current_label_id}...")
            
            if 'Vertex' not in group:
                print(f"      - Skipping, 'Vertex' dataset not found.")
                continue
                
            vertices = group['Vertex'][:]
            if len(vertices) == 0:
                print("      - Skipping, no vertices found.")
                continue

            vertices_drawn_count = 0
            
            # --- NEW DEBUG PRINT ---
            print_debug = True # Print first vertex of each filament
            # --- END DEBUG PRINT ---

            for v_idx, vertex in enumerate(vertices):
                # Assume PositionX/Y/Z are Voxel Coordinates
                vox_x = vertex['PositionX']
                vox_y = vertex['PositionY']
                vox_z = vertex['PositionZ']
                
                # --- NEW RADIUS LOGIC ---
                # Radius is in X/Y Pixels. We must scale it for Z.
                radius_in_xy_pixels = vertex['Radius']
                
                # Calculate Z/X and Y/X aspect ratios
                # vox_size is (Z, Y, X)
                # We assume Y and X are the same resolution, so Y/X aspect is 1
                z_to_x_aspect = vox_size[2] / vox_size[0] # e.g., 0.0039 / 0.0069 = ~0.56
                y_to_x_aspect = vox_size[2] / vox_size[1] # e.g., 0.0039 / 0.0039 = ~1.0
                
                radius_vox_x = radius_in_xy_pixels
                radius_vox_y = radius_in_xy_pixels * y_to_x_aspect # Should be same as X
                radius_vox_z = radius_in_xy_pixels * z_to_x_aspect # This will be the smaller value

                # Ensure radii are at least 1 pixel
                radius_vox_z = max(radius_vox_z, 1.0)
                radius_vox_y = max(radius_vox_y, 1.0)
                radius_vox_x = max(radius_vox_x, 1.0)
                # --- END NEW RADIUS LOGIC ---


                # --- NEW DEBUG PRINT ---
                if print_debug:
                    print(f"      - DEBUG Vertex {v_idx}:")
                    print(f"      -   Coords (px): (Z:{vox_z:.1f}, Y:{vox_y:.1f}, X:{vox_x:.1f})")
                    print(f"      -   Radius (in px_XY): {radius_in_xy_pixels:.2f}")
                    print(f"      -   Radius (scaled px): (Z:{radius_vox_z:.1f}, Y:{radius_vox_y:.1f}, X:{radius_vox_x:.1f})")
                    print_debug = False # Only print first one
                # --- END DEBUG PRINT ---

                # Check if vertex is inside the volume bounds
                v_z_int, v_y_int, v_x_int = int(vox_z), int(vox_y), int(vox_x)
                if (0 <= v_z_int < raw_shape[0] and
                    0 <= v_y_int < raw_shape[1] and
                    0 <= v_x_int < raw_shape[2]):
                    
                    # Draw this vertex as a sphere
                    draw_sphere_to_zarr(
                        label_volume_zarr, # Pass the Zarr array object
                        (vox_z, vox_y, vox_x), 
                        (radius_vox_z, radius_vox_y, radius_vox_x), 
                        current_label_id,
                        v_idx # Pass vertex index for debug prints
                    )
                    vertices_drawn_count += 1
                else:
                    pass 
            
            print(f"      - Done. Drew {vertices_drawn_count} / {len(vertices)} vertices for this filament.")
            if vertices_drawn_count > 0:
                total_vertices_drawn += vertices_drawn_count
                current_label_id += 1 
            else:
                print(f"      - Warning: No vertices for {group.name} were in bounds.")

        print("\n  --- RASTERIZATION COMPLETE ---")
        print(f"  Total vertices drawn: {total_vertices_drawn}")
        if total_vertices_drawn == 0:
            print("  🔴🔴🔴 ERROR: Drew 0 total vertices. Zarr file will be empty.")
        
        print(f"✅ Successfully saved Zarr file to: {zarr_filepath}")


def visualize_skeletons(ims_filepath, html_filepath):
    """
    Reads 'neuron.ims' and creates an interactive 3D plot
    of the filament skeletons, saved to 'neuron_skeletons.html'.
    """
    print(f"\n--- 2. Starting Visualization: {ims_filepath} -> {html_filepath} ---")
    
    with h5py.File(ims_filepath, 'r') as f:
        
        filament_groups = find_filament_groups(f) # Now recursive
        if not filament_groups:
            print("  No filament groups found. Aborting visualization.")
            return

        fig = go.Figure()
        
        # Process each filament group
        for i, group in enumerate(filament_groups):
            print(f"    - Plotting {group.name}...")
            if 'Vertex' not in group or 'Edge' not in group:
                print(f"      - Skipping, 'Vertex' or 'Edge' not found in {group.name}.")
                continue
            
            vertices = group['Vertex'][:]
            edges = group['Edge'][:]
            
            if len(vertices) == 0 or len(edges) == 0:
                print(f"      - Skipping, no vertices or edges found in {group.name}.")
                continue
            
            # Get all vertex positions
            pos_x = vertices['PositionX']
            pos_y = vertices['PositionY']
            pos_z = vertices['PositionZ']
            
            # Build the line segments for Plotly
            line_x, line_y, line_z = [], [], []
            
            for edge in edges:
                v_a_idx = edge['VertexA']
                v_b_idx = edge['VertexB']

                if v_a_idx < len(vertices) and v_b_idx < len(vertices):
                    v_a = vertices[v_a_idx]
                    v_b = vertices[v_b_idx]
                    
                    line_x.extend([v_a['PositionX'], v_b['PositionX'], None])
                    # --- TYPO FIX ---
                    # Was v.b['PositionY'], now v_b['PositionY']
                    line_y.extend([v_a['PositionY'], v_b['PositionY'], None])
                    line_z.extend([v_a['PositionZ'], v_b['PositionZ'], None])
                    # --- END FIX ---
                else:
                    print(f"      - Warning: Bad edge index in {group.name}. Skipping edge.")

            fig.add_trace(go.Scatter3d(
                x=line_x,
                y=line_y,
                z=line_z,
                mode='lines',
                line=dict(width=2),
                name=f"{os.path.basename(group.name)} (Edges)"
            ))

        print("  Finalizing 3D plot...")
        fig.update_layout(
            title="Interactive Filament Skeletons (Voxel Coordinates)",
            scene=dict(
                xaxis_title="X (pixels)",
                yaxis_title="Y (pixels)",
                zaxis_title="Z (pixels)",
                aspectmode='data' 
            )
        )
        
        fig.write_html(html_filepath)
        print(f"✅ Successfully saved 3D plot to: {html_filepath}")


if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("Usage: python process_filaments.py \"/path/to/your/file.ims\"")
        print("ℹ️ Remember to use quotes if the path contains spaces!")
        sys.exit(1)
        
    ims_file = sys.argv[1]

    # Derive output names from input name
    base_name = os.path.splitext(os.path.basename(ims_file))[0]
    zarr_file = f"{base_name}_labels.zarr"
    html_file = f"{base_name}_skeletons.html"
    
    if not os.path.exists(ims_file):
        print(f"FATAL ERROR: Input file not found: {ims_file}")
        sys.exit(1)
        
    try:
        rasterize_to_zarr(ims_file, zarr_file)
        visualize_skeletons(ims_file, html_file)
        
        print("\n--- All tasks complete ---")
        
    except Exception as e:
        print(f"\n--- A FATAL ERROR OCCURRED ---")
        print(f"{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)