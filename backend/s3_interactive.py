#!/usr/bin/env python3
"""
Interactive S3 Data Manager CLI

Walk through the user experience step by step:
  1. Create a project (or pick an existing one)
  2. Upload 3D data
  3. Choose a region to analyze
  4. Receive the inference result (saved as .npy file)
"""

import sys
import os
import numpy as np
import zarr
from s3_data_manager import S3DataManager


def prompt(msg, default=None):
    """Prompt user for input with optional default."""
    if default is not None:
        val = input(f"{msg} [{default}]: ").strip()
        return val if val else str(default)
    return input(f"{msg}: ").strip()


def prompt_int(msg, default=None):
    """Prompt for an integer."""
    while True:
        val = prompt(msg, default)
        try:
            return int(val)
        except ValueError:
            print("  Please enter a number.")


def print_header(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print()


def step_pause(msg="Press Enter to continue..."):
    input(f"\n{msg}")


# ======================================================================
# Steps
# ======================================================================

def step_connect(bucket):
    """Step 0: Connect to S3."""
    print_header("Step 0: Connect to S3")
    print(f"Connecting to bucket: {bucket}")
    mgr = S3DataManager(bucket)
    print("✓ Connected!")
    return mgr


def step_user(mgr):
    """Step 1: Identify yourself."""
    print_header("Step 1: Who are you?")
    user_id = prompt("Enter your user ID", "demo_user")

    # Show existing projects
    projects = mgr.list_projects(user_id)
    if projects:
        print(f"\nYou have {len(projects)} existing project(s):")
        for i, p in enumerate(projects):
            vol = f"shape={p['volume_shape']}" if p["volume_uploaded"] else "no volume"
            print(f"  [{i + 1}] {p['name']} ({p['project_id']}) — {vol}")

        print(f"  [0] Create new project")
        choice = prompt_int("\nPick a project or 0 for new", 0)

        if 1 <= choice <= len(projects):
            project = projects[choice - 1]
            print(f"\n✓ Using project: {project['name']}")
            return user_id, project

    return user_id, None


def step_create_project(mgr, user_id):
    """Step 2: Create a new project."""
    print_header("Step 2: Create a Project")
    name = prompt("Project name", "My Brain Scan")
    desc = prompt("Description (optional)", "")

    print(f"\nVoxel size (nanometers):")
    vx = prompt_int("  X", 4)
    vy = prompt_int("  Y", 4)
    vz = prompt_int("  Z", 30)

    project = mgr.create_project(
        user_id=user_id,
        name=name,
        description=desc,
        voxel_size_nm=[vx, vy, vz],
    )
    return project


def step_upload(mgr, project):
    """Step 3: Upload 3D volume data."""
    print_header("Step 3: Upload 3D Volume")

    print("Data source:")
    print("  [1] Generate dummy data (random 3D array)")
    print("  [2] Load from .npy file")
    choice = prompt_int("Choice", 1)

    if choice == 2:
        path = prompt("Path to .npy file")
        if not os.path.exists(path):
            print(f"✗ File not found: {path}")
            print("  Falling back to dummy data.")
            choice = 1

    if choice == 1:
        print("\nDummy volume dimensions:")
        sx = prompt_int("  X size", 200)
        sy = prompt_int("  Y size", 200)
        sz = prompt_int("  Z size", 200)
        print(f"\nGenerating random volume ({sx}x{sy}x{sz})...")
        data = np.random.randn(sx, sy, sz).astype(np.float32)
    else:
        print(f"\nLoading {path}...")
        data = np.load(path)

    size_mb = data.nbytes / (1024 * 1024)
    print(f"Volume: shape={data.shape}, dtype={data.dtype}, size={size_mb:.1f} MB")

    confirm = prompt(f"\nUpload to S3?", "y")
    if confirm.lower() != "y":
        print("Skipped upload.")
        return False

    print()
    mgr.upload_volume(project, data)
    return True


def step_choose_region(mgr, project):
    """Step 4: Choose a region to analyze or download an existing result."""
    print_header("Step 4: Analyze a Region")

    user_id = project["user_id"]
    project_id = project["project_id"]
    shape = project.get("volume_shape")
    region_size = list(mgr.region_size)

    print(f"Your volume: {shape}")
    print(f"Analysis region size: {region_size} (fixed)")

    # Show existing jobs
    jobs = mgr.list_jobs(user_id, project_id)
    if jobs:
        print(f"\nExisting jobs:")
        for i, j in enumerate(jobs):
            origin = j["region_origin"]
            status = j["status"]
            print(f"  [{i + 1}] origin={origin}  [{status}]  {j['job_id']}")
        print(f"  [0] Submit new region")

        choice = prompt_int("\nPick an existing job or 0 for new", 0)

        if 1 <= choice <= len(jobs):
            return jobs[choice - 1]

    # New job
    print()
    max_origin = [s - r for s, r in zip(shape, region_size)]
    print(f"Valid origin ranges:")
    print(f"  X: 0 to {max_origin[0]}")
    print(f"  Y: 0 to {max_origin[1]}")
    print(f"  Z: 0 to {max_origin[2]}")
    print()

    print("Enter the corner of the region you want analyzed:")
    x = prompt_int(f"  X origin (0–{max_origin[0]})", 0)
    y = prompt_int(f"  Y origin (0–{max_origin[1]})", 0)
    z = prompt_int(f"  Z origin (0–{max_origin[2]})", 0)

    origin = [x, y, z]
    print(f"\nRegion: origin={origin}, size={region_size}")
    print(f"This covers [{x}:{x+region_size[0]}, {y}:{y+region_size[1]}, {z}:{z+region_size[2]}]")

    confirm = prompt("\nSubmit this job?", "y")
    if confirm.lower() != "y":
        print("Cancelled.")
        return None

    print()
    project = mgr.get_project(user_id, project_id)
    job = mgr.submit_job(project, region_origin=origin)
    return job


def step_get_result(mgr, project, job):
    """Step 5: Download and save the result."""
    print_header("Step 5: Retrieve Your Result")

    user_id = project["user_id"]
    project_id = project["project_id"]
    job_id = job["job_id"]

    # Show job info
    job_info = mgr.get_job(user_id, project_id, job_id)
    print(f"Job:    {job_id}")
    print(f"Status: {job_info['status']}")
    print(f"Region: origin={job_info['region_origin']}, size={job_info['region_size']}")
    print()

    if job_info["status"] != "completed":
        print("⚠️  Job is not completed yet. Status:", job_info["status"])
        return

    # Download result
    result = mgr.download_result(user_id, project_id, job_id)

    neuron_count = np.sum(result == 1)
    total = result.size
    pct = 100 * neuron_count / total

    print(f"\nResult summary:")
    print(f"  Shape: {result.shape}")
    print(f"  Dtype: {result.dtype}")
    print(f"  Neuron voxels: {neuron_count:,} / {total:,} ({pct:.1f}%)")

    # Save to file
    default_name = f"result_{job_id}.zarr"
    save_path = prompt(f"\nSave result as", default_name)

    zarr.save(save_path, result)
    # Calculate folder size
    total_kb = 0
    for root, dirs, files in os.walk(save_path):
        for f in files:
            total_kb += os.path.getsize(os.path.join(root, f))
    total_kb /= 1024
    print(f"✓ Saved to {save_path}/ ({total_kb:.1f} KB)")


def step_more_jobs(mgr, project):
    """Ask if user wants to analyze another region or download another result."""
    print()
    again = prompt("Analyze or download another region?", "n")
    if again.lower() == "y":
        job = step_choose_region(mgr, project)
        if job:
            step_get_result(mgr, project, job)
            step_more_jobs(mgr, project)


def step_cleanup(mgr, project):
    """Optional cleanup."""
    print_header("Done!")

    user_id = project["user_id"]
    project_id = project["project_id"]

    # Show summary
    jobs = mgr.list_jobs(user_id, project_id)
    print(f"Project: {project['name']} ({project_id})")
    print(f"Jobs completed: {len(jobs)}")
    for j in jobs:
        print(f"  {j['job_id']} — origin={j['region_origin']} — {j['status']}")

    print()
    cleanup = prompt("Delete project from S3?", "n")
    if cleanup.lower() == "y":
        mgr.delete_project(user_id, project_id)
    else:
        print(f"\nYour data is at:")
        print(f"  s3://{mgr.bucket}/projects/{user_id}/{project_id}/")

    print("\n✓ All done!")


# ======================================================================
# Main
# ======================================================================

def main():
    print_header("S3 3D Data Manager — Interactive Mode")

    # Bucket
    if len(sys.argv) > 1:
        bucket = sys.argv[1]
    else:
        bucket = prompt("S3 bucket name", "my-3d-data-test-bucket-12345")

    try:
        # Connect
        mgr = step_connect(bucket)
        step_pause()

        # User + project selection
        user_id, project = step_user(mgr)

        # Create new project if needed
        if project is None:
            project = step_create_project(mgr, user_id)
            step_pause()

            # Upload volume
            uploaded = step_upload(mgr, project)
            if not uploaded:
                print("Cannot continue without uploading data.")
                return
        else:
            if not project.get("volume_uploaded"):
                print("\nThis project has no volume yet.")
                uploaded = step_upload(mgr, project)
                if not uploaded:
                    print("Cannot continue without uploading data.")
                    return

        # Refresh project metadata
        project = mgr.get_project(user_id, project["project_id"])
        step_pause()

        # Choose region + submit job
        job = step_choose_region(mgr, project)
        if job is None:
            print("No job submitted.")
            return

        step_pause()

        # Get result
        step_get_result(mgr, project, job)

        # More jobs?
        step_more_jobs(mgr, project)

        # Cleanup
        step_cleanup(mgr, project)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted. Bye!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()