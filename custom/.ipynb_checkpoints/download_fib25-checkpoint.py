import boto3
import os
import sys
from botocore import UNSIGNED
from botocore.client import Config

# --- Helper Functions (from our previous scripts) ---

def list_zarr_directories(bucket, prefix):
    """Lists directories ending in .zarr under a given prefix."""
    print(f"Listing Zarr directories in: {prefix}")
    paginator = client.get_paginator('list_objects_v2')
    zarr_folders = set()  # <-- Variable is defined here
    
    if not prefix.endswith('/'):
        prefix += '/'
    
    try:
        # Use Delimiter='/' to find top-level "folders"
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter='/'):
            if "CommonPrefixes" in page:
                for p in page['CommonPrefixes']:
                    full_prefix = p['Prefix']
                    if full_prefix.endswith('.zarr/'):
                        zarr_folders.add(full_prefix)
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return sorted(list(zarr_folders))  # <-- This line is now fixed

def downloadDirectory(bucket_name, path):
    """Downloads all files from a given S3 path, skipping existing files."""
    resource = boto3.resource(
        's3',
        config=Config(signature_version=UNSIGNED)
    )
    bucket = resource.Bucket(bucket_name)

    print(f"\n--- Starting download for: {path} ---")
    
    # Check if any objects exist at this prefix
    objects = list(bucket.objects.filter(Prefix=path).limit(1))
    if not objects:
        print(f"Warning: No objects found at path {path}. Skipping.")
        return

    for obj in bucket.objects.filter(Prefix=path):
        # Create local directory structure
        if not os.path.exists(os.path.dirname(obj.key)):
            os.makedirs(os.path.dirname(obj.key))
        
        key = obj.key
        
        # Check if file already exists
        if not os.path.exists(key):
            print(f'Downloading {key}')
            bucket.download_file(key, key)
        else:
            print(f'Skipping {key} (already exists)')
    print(f"--- Finished download for: {path} ---")

# --- Main part of the script ---

# 1. Connect to S3
client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
bucket_name = 'open-neurodata'
fib25_training_prefix = 'funke/fib25/training/'

# 2. Find ONLY the Zarr files
print(f"Finding Zarr files in {fib25_training_prefix}")
print("(This will ignore the 'checkpoints' folder and any other files)")
zarr_list = list_zarr_directories(bucket_name, fib25_training_prefix)

if not zarr_list:
    print(f"Error: No .zarr directories found in {fib25_training_prefix}")
    sys.exit(1)

print(f"\nFound {len(zarr_list)} Zarr files (the ~297 MB of source data):")
for zarr_name in zarr_list:
    print(f"- {zarr_name}")

# 3. Ask for confirmation
# We know the approximate size from our previous analysis
print("\nTotal download size will be approximately 297 MB.")
print(f"This will save you ~22.6 GB by skipping the 'checkpoints' folder.")

try:
    proceed = input("Do you want to proceed with the download? (y/n): ")
except EOFError:
    proceed = 'n'
    print("Non-interactive mode. Exiting.")

if proceed.lower() == 'y':
    # 4. Loop and download EACH zarr file individually
    print("\nStarting selective download...")
    for zarr_to_download in zarr_list:
        downloadDirectory(bucket_name, zarr_to_download)
    
    print("\n--- All Zarr file downloads complete ---")
    print("The 'checkpoints' folder was successfully skipped.")

else:
    print("Download cancelled.")