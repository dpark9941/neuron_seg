import boto3
import os
from botocore import UNSIGNED
from botocore.client import Config

def get_s3_folder_size(bucket, prefix):
    """Calculates the total size of a 'folder' in an S3 bucket."""
    total_size = 0
    paginator = client.get_paginator('list_objects_v2')
    
    if not prefix.endswith('/'):
        prefix += '/'
    
    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if "Contents" in page:
                for obj in page['Contents']:
                    total_size += obj['Size']
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

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

def list_all_directories(bucket, prefix):
    """Lists all top-level 'folders' under a given prefix."""
    print(f"Listing all directories in: {prefix}")
    paginator = client.get_paginator('list_objects_v2')
    folders = set()
    
    if not prefix.endswith('/'):
        prefix += '/'
    
    try:
        # Use Delimiter='/' to find top-level "folders"
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter='/'):
            if "CommonPrefixes" in page:
                for p in page['CommonPrefixes']:
                    folders.add(p['Prefix'])
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return sorted(list(folders))

# --- Main part of the script ---
client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
bucket_name = 'open-neurodata'
fib25_training_prefix = 'funke/fib25/training/'

# 1. List ALL directories in the training folder
all_dirs = list_all_directories(bucket_name, fib25_training_prefix)
print(f"Found {len(all_dirs)} top-level directories.\n")

total_size_check = 0
total_size_mb = 0

# 2. Get the size of each directory
for dir_prefix in all_dirs:
    print(f"--- Analyzing: {dir_prefix} ---")
    size_str = get_s3_folder_size(bucket_name, dir_prefix)
    print(f"  Total Size: {size_str}\n")
    
    # Track the total to see if it adds up
    if 'GB' in size_str:
        total_size_mb += float(size_str.split()[0]) * 1024
    elif 'MB' in size_str:
        total_size_mb += float(size_str.split()[0])

print(f"--- Summary ---")
print(f"Calculated total size: {total_size_mb / 1024:.2f} GB")
print(f"Original script reported: 22.89 GB")