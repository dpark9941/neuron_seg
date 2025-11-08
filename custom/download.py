import boto3
from botocore import UNSIGNED
from botocore.client import Config

def get_s3_folder_size(bucket, prefix):
    """Calculates the total size of a 'folder' in an S3 bucket."""
    total_size = 0
    paginator = client.get_paginator('list_objects_v2')
    
    # Ensure the prefix ends with a '/' to treat it as a folder
    if not prefix.endswith('/'):
        prefix += '/'

    print(f"Calculating size for: {prefix}...")
    
    try:
        # Check if the prefix itself exists by trying to list it
        result = client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        if 'Contents' not in result and 'CommonPrefixes' not in result:
             # If there are no contents and no common prefixes, the folder is empty or doesn't exist
             print(f"Result: 0.00 MB (Path may be empty or incorrect)")
             return "0.00 MB"

        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if "Contents" in page:
                for obj in page['Contents']:
                    total_size += obj['Size']
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    # Convert bytes to a human-readable format
    if total_size == 0:
        return "0.00 MB"
    elif total_size < 1024**2:
        return f"{total_size / 1024.0:.2f} KB"
    elif total_size < 1024**3:
        return f"{total_size / (1024**2):.2f} MB"
    elif total_size < 1024**4:
        return f"{total_size / (1024**3):.2f} GB"
    else:
        return f"{total_size / (1024**4):.2f} TB"

def list_zarr_directories(bucket, prefix):
    """Lists directories ending in .zarr under a given prefix."""
    print(f"\nListing Zarr directories in: {prefix}")
    paginator = client.get_paginator('list_objects_v2')
    zarr_folders = set()
    
    if not prefix.endswith('/'):
        prefix += '/'
    
    try:
        # Use Delimiter='/' to find top-level "folders"
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter='/'):
            if "CommonPrefixes" in page:
                for p in page['CommonPrefixes']:
                    # Get the folder name, e.g., 'funke/fib25/training/trvol-250-1.zarr/'
                    full_prefix = p['Prefix']
                    if full_prefix.endswith('.zarr/'):
                        zarr_folders.add(full_prefix)
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return sorted(list(zarr_folders))

# --- Main part of the script ---

# 1. Connect to S3
client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
bucket_name = 'open-neurodata'

# 2. Find all FIB-25 training Zarrs
fib25_training_prefix = 'funke/fib25/training/'
fib25_zarr_list = list_zarr_directories(bucket_name, fib25_training_prefix)

print(f"Found {len(fib25_zarr_list)} Zarr files in {fib25_training_prefix}:")
for zarr_name in fib25_zarr_list:
    print(f"- {zarr_name}")

# 3. Pick one Zarr file and calculate the size of its components
if fib25_zarr_list:
    # Let's use the first one as an example
    example_zarr_prefix = fib25_zarr_list[0] 
    
    print(f"\n--- Calculating component sizes for {example_zarr_prefix} ---")

    # A) Calculate size of the 'raw' data
    raw_prefix = f"{example_zarr_prefix}volumes/raw/"
    raw_size = get_s3_folder_size(bucket_name, raw_prefix)
    print(f"Size of 'raw' data ({raw_prefix}): {raw_size}")
    
    # B) Calculate size of the 'labels' data (neuron_ids)
    # Based on your <a name="s3-array"> example, the data is in 'volumes/labels/neuron_ids'
    labels_prefix = f"{example_zarr_prefix}volumes/labels/neuron_ids/"
    labels_size = get_s3_folder_size(bucket_name, labels_prefix)
    print(f"Size of 'labels' (neuron_ids) data ({labels_prefix}): {labels_size}")
    
else:
    print(f"No .zarr directories found in {fib25_training_prefix}")

# 4. (Optional) Loop and calculate 'raw' and 'labels' size for ALL 4 FIB-25 Zarrs
print("\n--- Calculating component sizes for all FIB-25 training Zarrs ---")
total_raw_size_bytes = 0
total_labels_size_bytes = 0

for zarr_prefix in fib25_zarr_list:
    print(f"Analyzing: {zarr_prefix}")
    
    # Get raw size
    raw_prefix = f"{zarr_prefix}volumes/raw/"
    raw_size_str = get_s3_folder_size(bucket_name, raw_prefix)
    print(f"  > Raw size: {raw_size_str}")
    
    # Get labels size
    labels_prefix = f"{zarr_prefix}volumes/labels/neuron_ids/"
    labels_size_str = get_s3_folder_size(bucket_name, labels_prefix)
    print(f"  > Labels (neuron_ids) size: {labels_size_str}")