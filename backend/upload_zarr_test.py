#!/usr/bin/env python3
"""
S3 Zarr 3D Data Test Script (Zarr v3 compatible)
Tests uploading and downloading 3D zarr arrays to/from S3
"""

import boto3
import s3fs
import zarr
import numpy as np
import sys
import time
from pathlib import Path

# Configuration
BUCKET_NAME = "my-3d-data-test-bucket-12345"  # CHANGE THIS
S3_PATH = f"s3://{BUCKET_NAME}/test-zarr-data"

def create_sample_3d_data():
    """Create a sample 3D array (simulating 3D scan data)"""
    print("\n=== Creating Sample 3D Data ===")
    
    # Create a 100x100x100 array with some interesting pattern
    # This simulates a small 3D scan
    shape = (100, 100, 100)
    
    print(f"Creating {shape} array...")
    
    # Create data with a 3D sphere pattern
    x, y, z = np.ogrid[-50:50, -50:50, -50:50]
    data = np.sqrt(x**2 + y**2 + z**2)
    data = 255 * (data < 30).astype(np.float32)  # Sphere with radius 30
    
    # Add some noise
    data += np.random.randn(*shape).astype(np.float32) * 10
    
    size_mb = data.nbytes / (1024 * 1024)
    print(f"✓ Created 3D array: shape={shape}, dtype={data.dtype}")
    print(f"✓ Uncompressed size: {size_mb:.2f} MB")
    
    return data

def upload_zarr_to_s3(data, s3_path):
    """Upload 3D data as zarr to S3 (Zarr v3 compatible)"""
    print(f"\n=== Uploading Zarr to S3 ===")
    print(f"Destination: {s3_path}")
    
    try:
        start_time = time.time()
        
        # Create S3 filesystem
        s3 = s3fs.S3FileSystem()
        
        # Create zarr store on S3
        store = s3fs.S3Map(root=s3_path, s3=s3, check=False)
        
        # Create zarr array with compression (v3 syntax)
        print("Writing zarr array with compression...")
        
        # Check zarr version
        zarr_version = int(zarr.__version__.split('.')[0])
        
        if zarr_version >= 3:
            # Zarr v3 syntax - use codecs
            from numcodecs import Blosc
            
            z = zarr.open(
                store,
                mode='w',
                shape=data.shape,
                chunks=(50, 50, 50),
                dtype=data.dtype,
                zarr_format=2,  # Use format 2 for compatibility
                compressor=Blosc(cname='zstd', clevel=3)
            )
        else:
            # Zarr v2 syntax
            z = zarr.open(
                store,
                mode='w',
                shape=data.shape,
                chunks=(50, 50, 50),
                dtype=data.dtype,
                compressor=zarr.Blosc(cname='zstd', clevel=3)
            )
        
        # Write data
        z[:] = data
        
        elapsed = time.time() - start_time
        print(f"✓ Upload completed in {elapsed:.2f} seconds")
        
        # Check compression ratio
        print("\nChecking S3 storage...")
        total_size = 0
        for obj in s3.ls(s3_path, detail=True):
            total_size += obj['size']
        
        compressed_mb = total_size / (1024 * 1024)
        original_mb = data.nbytes / (1024 * 1024)
        ratio = original_mb / compressed_mb if compressed_mb > 0 else 0
        
        print(f"✓ Original size: {original_mb:.2f} MB")
        print(f"✓ Compressed size on S3: {compressed_mb:.2f} MB")
        print(f"✓ Compression ratio: {ratio:.2f}x")
        
        return True
    
    except Exception as e:
        print(f"✗ Upload failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def download_zarr_from_s3(s3_path):
    """Download and read zarr data from S3"""
    print(f"\n=== Downloading Zarr from S3 ===")
    print(f"Source: {s3_path}")
    
    try:
        start_time = time.time()
        
        # Create S3 filesystem
        s3 = s3fs.S3FileSystem()
        
        # Open zarr store from S3
        store = s3fs.S3Map(root=s3_path, s3=s3, check=False)
        
        print("Opening zarr array...")
        z = zarr.open(store, mode='r')
        
        print(f"✓ Opened zarr array: shape={z.shape}, dtype={z.dtype}")
        print(f"✓ Chunks: {z.chunks}")
        
        # Read a subset (don't download everything)
        print("\nReading a subset (first 50x50x50 cube)...")
        subset = z[0:50, 0:50, 0:50]
        
        elapsed = time.time() - start_time
        print(f"✓ Downloaded subset in {elapsed:.2f} seconds")
        print(f"✓ Subset shape: {subset.shape}")
        
        # Now read the entire array
        print("\nReading entire array...")
        start_full = time.time()
        full_data = z[:]
        elapsed_full = time.time() - start_full
        
        print(f"✓ Downloaded full array in {elapsed_full:.2f} seconds")
        
        return full_data
    
    except Exception as e:
        print(f"✗ Download failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def verify_data_integrity(original, downloaded):
    """Verify the downloaded data matches the original"""
    print("\n=== Verifying Data Integrity ===")
    
    if downloaded is None:
        print("✗ No data to verify")
        return False
    
    # Check shape
    if original.shape != downloaded.shape:
        print(f"✗ Shape mismatch: {original.shape} vs {downloaded.shape}")
        return False
    print(f"✓ Shape matches: {original.shape}")
    
    # Check dtype
    if original.dtype != downloaded.dtype:
        print(f"✗ Dtype mismatch: {original.dtype} vs {downloaded.dtype}")
        return False
    print(f"✓ Dtype matches: {original.dtype}")
    
    # Check values (allowing for floating point errors)
    if np.allclose(original, downloaded):
        print("✓ Values match (within tolerance)")
        max_diff = np.max(np.abs(original - downloaded))
        print(f"✓ Max difference: {max_diff:.2e}")
        return True
    else:
        print("✗ Values don't match!")
        max_diff = np.max(np.abs(original - downloaded))
        print(f"  Max difference: {max_diff}")
        return False

def cleanup_s3(s3_path):
    """Clean up S3 files"""
    print(f"\n=== Cleanup ===")
    
    try:
        s3 = s3fs.S3FileSystem()
        
        # Delete all files in the zarr directory
        if s3.exists(s3_path):
            s3.rm(s3_path, recursive=True)
            print(f"✓ Deleted from S3: {s3_path}")
        else:
            print(f"⚠️  Path not found: {s3_path}")
    
    except Exception as e:
        print(f"✗ Cleanup error: {e}")

def test_chunked_access(s3_path):
    """Test reading only specific chunks (memory efficient)"""
    print(f"\n=== Testing Chunked Access ===")
    print("(This is the key advantage of zarr for large 3D data)")
    
    try:
        s3 = s3fs.S3FileSystem()
        store = s3fs.S3Map(root=s3_path, s3=s3, check=False)
        z = zarr.open(store, mode='r')
        
        # Read different slices without loading entire array
        print("\nReading various slices...")
        
        # 2D slice (XY plane at z=50)
        start = time.time()
        slice_xy = z[:, :, 50]
        print(f"✓ XY slice (z=50): {slice_xy.shape} - {time.time()-start:.3f}s")
        
        # 2D slice (XZ plane at y=50)
        start = time.time()
        slice_xz = z[:, 50, :]
        print(f"✓ XZ slice (y=50): {slice_xz.shape} - {time.time()-start:.3f}s")
        
        # Small 3D cube
        start = time.time()
        cube = z[25:75, 25:75, 25:75]
        print(f"✓ Central cube: {cube.shape} - {time.time()-start:.3f}s")
        
        print("\n✓ Chunked access working! Only downloads needed data.")
        return True
    
    except Exception as e:
        print(f"✗ Chunked access failed: {e}")
        return False

def main():
    print("=" * 60)
    print("AWS S3 Zarr 3D Data Test Script")
    #print(f"Zarr version: {zarr.__version__}")
    print("=" * 60)
    
    # Check configuration
    global BUCKET_NAME, S3_PATH
    if BUCKET_NAME == "your-bucket-name":
        if len(sys.argv) > 1:
            BUCKET_NAME = sys.argv[1]
            S3_PATH = f"s3://{BUCKET_NAME}/test-zarr-data"
            print(f"Using bucket: {BUCKET_NAME}")
        else:
            print("\n⚠️  Please provide bucket name:")
            print("   python test_s3_zarr.py your-bucket-name")
            sys.exit(1)
    
    # Check dependencies
    try:
        import s3fs
        import zarr
        import numcodecs
    except ImportError as e:
        print(f"\n✗ Missing dependency: {e}")
        print("\nPlease install:")
        print("  pip install s3fs zarr numpy numcodecs")
        sys.exit(1)
    
    try:
        # 1. Create sample 3D data
        original_data = create_sample_3d_data()
        
        # 2. Upload as zarr to S3
        success = upload_zarr_to_s3(original_data, S3_PATH)
        if not success:
            sys.exit(1)
        
        # 3. Test chunked access
        test_chunked_access(S3_PATH)
        
        # 4. Download full data
        downloaded_data = download_zarr_from_s3(S3_PATH)
        
        # 5. Verify integrity
        verified = verify_data_integrity(original_data, downloaded_data)
        
        # 6. Cleanup
        print("\nDo you want to clean up test data from S3? (y/n): ", end='')
        response = input().lower()
        if response == 'y':
            cleanup_s3(S3_PATH)
        else:
            print(f"\nTest data left in S3: {S3_PATH}")
            print("You can view it with: aws s3 ls s3://{}/test-zarr-data/".format(BUCKET_NAME))
        
        # Final summary
        print("\n" + "=" * 60)
        if verified:
            print("✓ ALL TESTS PASSED!")
            print("✓ Your S3 + Zarr setup is working correctly!")
            print("✓ You can now use this for large 3D data!")
        else:
            print("⚠️  Tests completed with errors")
        print("=" * 60)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()