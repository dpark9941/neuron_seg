#!/usr/bin/env python3
"""
S3 Bucket Inspector

Quick overview of what's in your bucket: projects, volumes, and jobs.

Usage:
    python s3_inspect.py <bucket-name>
    python s3_inspect.py my-3d-data-test-bucket-12345
"""

import sys
import json
import s3fs


def inspect_bucket(bucket):
    s3 = s3fs.S3FileSystem()

    print()
    print(f"🪣 Bucket: {bucket}")
    print("=" * 60)

    # Check if projects/ exists
    projects_root = f"{bucket}/projects"
    try:
        user_dirs = s3.ls(projects_root, detail=False)
    except FileNotFoundError:
        print("\n  (empty — no projects found)")
        return

    total_size = 0

    for user_dir in sorted(user_dirs):
        user_id = user_dir.rstrip("/").split("/")[-1]
        print(f"\n👤 User: {user_id}")
        print("-" * 40)

        try:
            project_dirs = s3.ls(user_dir, detail=False)
        except FileNotFoundError:
            continue

        for proj_dir in sorted(project_dirs):
            project_id = proj_dir.rstrip("/").split("/")[-1]

            # Read project metadata
            meta_path = f"{proj_dir}/metadata.json"
            try:
                with s3.open(meta_path, "r") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}

            name = meta.get("name", "???")
            shape = meta.get("volume_shape")
            uploaded = meta.get("volume_uploaded", False)
            created = meta.get("created_at", "?")[:19]  # trim timezone

            # Calculate project size on S3
            proj_size = sum(
                obj["size"] for obj in s3.ls(proj_dir, detail=True, refresh=True)
                if obj["type"] == "file"
            )
            # Also check subdirs
            for item in s3.find(proj_dir, detail=True).values():
                proj_size = 0
            for item in s3.find(proj_dir, detail=True).values():
                proj_size += item.get("size", 0)

            total_size += proj_size
            size_str = _format_size(proj_size)

            print(f"\n  📁 {name}")
            print(f"     ID:      {project_id}")
            print(f"     Created: {created}")
            if uploaded and shape:
                print(f"     Volume:  {shape[0]}×{shape[1]}×{shape[2]}")
            else:
                print(f"     Volume:  (not uploaded)")
            print(f"     Size:    {size_str}")

            # List jobs
            jobs_root = f"{proj_dir}/jobs"
            try:
                job_dirs = s3.ls(jobs_root, detail=False)
            except FileNotFoundError:
                job_dirs = []

            if job_dirs:
                print(f"     Jobs:    {len(job_dirs)}")
                for job_dir in sorted(job_dirs):
                    job_id = job_dir.rstrip("/").split("/")[-1]
                    try:
                        with s3.open(f"{job_dir}/metadata.json", "r") as f:
                            jmeta = json.load(f)
                        origin = jmeta.get("region_origin", "?")
                        status = jmeta.get("status", "?")
                        print(f"               • {job_id}  origin={origin}  [{status}]")
                    except Exception:
                        print(f"               • {job_id}  (no metadata)")
            else:
                print(f"     Jobs:    (none)")

    print()
    print("=" * 60)
    print(f"Total S3 usage: {_format_size(total_size)}")
    print("=" * 60)


def _format_size(nbytes):
    if nbytes < 1024:
        return f"{nbytes} B"
    elif nbytes < 1024 * 1024:
        return f"{nbytes / 1024:.1f} KB"
    elif nbytes < 1024 * 1024 * 1024:
        return f"{nbytes / (1024 * 1024):.1f} MB"
    else:
        return f"{nbytes / (1024 * 1024 * 1024):.2f} GB"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("S3 Bucket Inspector")
        print("=" * 40)
        print()
        print("Usage:")
        print(f"  python {sys.argv[0]} <bucket-name>")
        print()
        print("Example:")
        print(f"  python {sys.argv[0]} my-3d-data-test-bucket-12345")
    else:
        inspect_bucket(sys.argv[1])