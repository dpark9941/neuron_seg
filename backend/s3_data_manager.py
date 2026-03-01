#!/usr/bin/env python3
"""
S3 Data Manager for 3D Zarr Volumes

Manages uploading 3D data, submitting analysis jobs (region requests),
and retrieving results from S3.

Structure:
    s3://bucket/projects/{user_id}/{project_id}/
        metadata.json
        raw/volume.zarr/
        jobs/{job_id}/
            metadata.json
            result.zarr/

Usage:
    from s3_data_manager import S3DataManager

    mgr = S3DataManager("my-bucket-name")

    # Upload a volume
    project = mgr.create_project("user_01", "Mouse Brain Sample 7",
                                  voxel_size_nm=[4, 4, 30])
    mgr.upload_volume(project, my_numpy_array, chunks=(128, 128, 128))

    # Submit a job (region to analyze)
    job = mgr.submit_job(project, region_origin=[512, 256, 100])

    # (cluster runs model, writes result...)
    # mgr.upload_result(project, job["job_id"], result_array)

    # Check status and retrieve
    job = mgr.get_job(project, job["job_id"])
    result = mgr.download_result(project, job["job_id"])
"""

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import s3fs
import zarr
from numcodecs import Blosc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_id(prefix: str = "") -> str:
    """Generate a short unique ID with optional prefix."""
    short = uuid.uuid4().hex[:12]
    return f"{prefix}_{short}" if prefix else short


def _now_iso() -> str:
    """Current UTC time as ISO string."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# S3DataManager
# ---------------------------------------------------------------------------

class S3DataManager:
    """
    Manages 3D zarr data on S3 organized by projects and analysis jobs.

    Parameters
    ----------
    bucket : str
        S3 bucket name.
    default_region_size : tuple[int, int, int]
        Default size of analysis regions (constant per your spec).
    compressor : optional
        Numcodecs compressor for zarr arrays. Defaults to Blosc/zstd.
    """

    # Default analysis region size (constant across all jobs)
    DEFAULT_REGION_SIZE = (128, 128, 128)

    def __init__(
        self,
        bucket: str,
        default_region_size: tuple[int, int, int] = DEFAULT_REGION_SIZE,
        compressor=None,
        simulate_inference: bool = True,
    ):
        self.bucket = bucket
        self.region_size = default_region_size
        self.compressor = compressor or Blosc(cname="zstd", clevel=3)
        self.simulate_inference = simulate_inference
        self.s3 = s3fs.S3FileSystem()

    # ------------------------------------------------------------------
    # Internal path helpers
    # ------------------------------------------------------------------

    def _project_root(self, user_id: str, project_id: str) -> str:
        return f"s3://{self.bucket}/projects/{user_id}/{project_id}"

    def _project_meta_path(self, user_id: str, project_id: str) -> str:
        return f"{self._project_root(user_id, project_id)}/metadata.json"

    def _volume_path(self, user_id: str, project_id: str) -> str:
        return f"{self._project_root(user_id, project_id)}/raw/volume.zarr"

    def _job_root(self, user_id: str, project_id: str, job_id: str) -> str:
        return f"{self._project_root(user_id, project_id)}/jobs/{job_id}"

    def _job_meta_path(self, user_id: str, project_id: str, job_id: str) -> str:
        return f"{self._job_root(user_id, project_id, job_id)}/metadata.json"

    def _result_path(self, user_id: str, project_id: str, job_id: str) -> str:
        return f"{self._job_root(user_id, project_id, job_id)}/result.zarr"

    # ------------------------------------------------------------------
    # Metadata I/O
    # ------------------------------------------------------------------

    def _write_meta(self, path: str, data: dict):
        """Write a JSON metadata file to S3."""
        with self.s3.open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _read_meta(self, path: str) -> dict:
        """Read a JSON metadata file from S3."""
        with self.s3.open(path, "r") as f:
            return json.load(f)

    # ==================================================================
    # PROJECT MANAGEMENT
    # ==================================================================

    def create_project(
        self,
        user_id: str,
        name: str,
        description: str = "",
        voxel_size_nm: Optional[list] = None,
    ) -> dict:
        """
        Create a new project (before uploading the volume).

        Parameters
        ----------
        user_id : str
            User identifier.
        name : str
            Human-readable project name.
        description : str
            Optional description.
        voxel_size_nm : list of 3 floats, optional
            Voxel dimensions in nanometers [x, y, z].

        Returns
        -------
        dict
            Project metadata (including project_id, user_id, paths).
        """
        project_id = _generate_id("proj")

        metadata = {
            "project_id": project_id,
            "user_id": user_id,
            "name": name,
            "description": description,
            "created_at": _now_iso(),
            "voxel_size_nm": voxel_size_nm,
            "volume_shape": None,       # filled after upload
            "volume_dtype": None,       # filled after upload
            "volume_uploaded": False,
        }

        self._write_meta(self._project_meta_path(user_id, project_id), metadata)
        print(f"✓ Created project '{name}' ({project_id})")
        return metadata

    def get_project(self, user_id: str, project_id: str) -> dict:
        """Retrieve project metadata."""
        return self._read_meta(self._project_meta_path(user_id, project_id))

    def list_projects(self, user_id: str) -> list[dict]:
        """List all projects for a user."""
        user_root = f"s3://{self.bucket}/projects/{user_id}/"
        projects = []

        try:
            project_dirs = self.s3.ls(f"{self.bucket}/projects/{user_id}", detail=False)
        except FileNotFoundError:
            return []

        for pdir in project_dirs:
            project_id = pdir.rstrip("/").split("/")[-1]
            try:
                meta = self.get_project(user_id, project_id)
                projects.append(meta)
            except Exception:
                continue

        return projects

    def delete_project(self, user_id: str, project_id: str):
        """Delete a project and all its data (volume + all jobs)."""
        root = self._project_root(user_id, project_id)
        if self.s3.exists(root):
            self.s3.rm(root, recursive=True)
            print(f"✓ Deleted project {project_id}")
        else:
            print(f"⚠️  Project not found: {project_id}")

    # ==================================================================
    # VOLUME UPLOAD / DOWNLOAD
    # ==================================================================

    def upload_volume(
        self,
        project: dict,
        data: np.ndarray,
        chunks: tuple[int, ...] = (128, 128, 128),
    ):
        """
        Upload a 3D numpy array as a zarr volume to S3.

        Parameters
        ----------
        project : dict
            Project metadata (from create_project).
        data : np.ndarray
            3D numpy array to upload.
        chunks : tuple
            Chunk sizes for zarr storage.
        """
        user_id = project["user_id"]
        project_id = project["project_id"]
        vol_path = self._volume_path(user_id, project_id)

        print(f"Uploading volume: shape={data.shape}, dtype={data.dtype}")
        print(f"  → {vol_path}")

        start = time.time()

        store = s3fs.S3Map(root=vol_path, s3=self.s3, check=False)
        z = zarr.open(
            store,
            mode="w",
            shape=data.shape,
            chunks=chunks,
            dtype=data.dtype,
            zarr_format=2,
            compressor=self.compressor,
        )
        z[:] = data

        elapsed = time.time() - start
        size_mb = data.nbytes / (1024 * 1024)
        print(f"✓ Uploaded {size_mb:.1f} MB in {elapsed:.2f}s")

        # Update project metadata with volume info
        project["volume_shape"] = list(data.shape)
        project["volume_dtype"] = str(data.dtype)
        project["volume_uploaded"] = True
        self._write_meta(
            self._project_meta_path(user_id, project_id), project
        )

    def open_volume(self, user_id: str, project_id: str):
        """
        Open a zarr volume from S3 (lazy, no data downloaded yet).

        Returns a zarr array — slice it to download only what you need.
        """
        vol_path = self._volume_path(user_id, project_id)
        store = s3fs.S3Map(root=vol_path, s3=self.s3, check=False)
        return zarr.open(store, mode="r")

    def download_volume(self, user_id: str, project_id: str) -> np.ndarray:
        """Download the entire volume as a numpy array."""
        print("Downloading full volume...")
        start = time.time()
        z = self.open_volume(user_id, project_id)
        data = z[:]
        elapsed = time.time() - start
        print(f"✓ Downloaded {data.shape} in {elapsed:.2f}s")
        return data

    def download_region(
        self,
        user_id: str,
        project_id: str,
        origin: list[int],
        size: Optional[list[int]] = None,
    ) -> np.ndarray:
        """
        Download a specific region from the volume (efficient chunked read).

        Parameters
        ----------
        origin : list of 3 ints
            Corner [x0, y0, z0] of the region.
        size : list of 3 ints, optional
            Region dimensions. Defaults to self.region_size.
        """
        size = size or list(self.region_size)
        z = self.open_volume(user_id, project_id)

        x0, y0, z0 = origin
        sx, sy, sz = size

        print(f"Downloading region: origin={origin}, size={size}")
        start = time.time()
        region = z[x0 : x0 + sx, y0 : y0 + sy, z0 : z0 + sz]
        elapsed = time.time() - start
        print(f"✓ Downloaded region {region.shape} in {elapsed:.2f}s")
        return region

    # ==================================================================
    # JOB MANAGEMENT
    # ==================================================================

    def submit_job(
        self,
        project: dict,
        region_origin: list[int],
        region_size: Optional[list[int]] = None,
    ) -> dict:
        """
        Submit an analysis job for a region in the volume.

        Parameters
        ----------
        project : dict
            Project metadata.
        region_origin : list of 3 ints
            Corner [x0, y0, z0] of the region to analyze.
        region_size : list of 3 ints, optional
            Size of the region. Defaults to self.region_size.

        Returns
        -------
        dict
            Job metadata (including job_id).
        """
        user_id = project["user_id"]
        project_id = project["project_id"]
        region_size = region_size or list(self.region_size)

        # Validate region is within volume bounds
        if project.get("volume_shape"):
            vol_shape = project["volume_shape"]
            for i, (o, s, v) in enumerate(zip(region_origin, region_size, vol_shape)):
                if o < 0 or o + s > v:
                    raise ValueError(
                        f"Region out of bounds on axis {i}: "
                        f"origin={o}, size={s}, volume={v}"
                    )

        job_id = _generate_id("job")

        metadata = {
            "job_id": job_id,
            "project_id": project_id,
            "user_id": user_id,
            "region_origin": region_origin,
            "region_size": region_size,
            "status": "queued",         # queued → running → completed / failed
            "submitted_at": _now_iso(),
            "started_at": None,
            "completed_at": None,
            "error": None,
        }

        self._write_meta(
            self._job_meta_path(user_id, project_id, job_id), metadata
        )
        print(f"✓ Submitted job {job_id}: origin={region_origin}, size={region_size}")

        # --- Simulate model inference (placeholder) ---
        # In production, set simulate_inference=False and let the cluster
        # worker pick up queued jobs, run the model, and call upload_result.
        if self.simulate_inference:
            print(f"  [SIM] Simulating inference for job {job_id}...")
            self.update_job_status(user_id, project_id, job_id, "running")

            # Fake classification: random 0/1 labels
            fake_result = np.random.randint(
                0, 2, size=tuple(region_size), dtype=np.uint8
            )
            self.upload_result(user_id, project_id, job_id, fake_result)
            print(f"  [SIM] Job {job_id} completed with simulated result")

            # Refresh metadata to return completed status
            metadata = self.get_job(user_id, project_id, job_id)

        return metadata

    def get_job(self, user_id: str, project_id: str, job_id: str) -> dict:
        """Retrieve job metadata."""
        return self._read_meta(self._job_meta_path(user_id, project_id, job_id))

    def list_jobs(
        self,
        user_id: str,
        project_id: str,
        status: Optional[str] = None,
    ) -> list[dict]:
        """
        List all jobs for a project, optionally filtered by status.

        Parameters
        ----------
        status : str, optional
            Filter by status: "queued", "running", "completed", "failed".
        """
        jobs_root = f"{self.bucket}/projects/{user_id}/{project_id}/jobs"
        jobs = []

        try:
            job_dirs = self.s3.ls(jobs_root, detail=False)
        except FileNotFoundError:
            return []

        for jdir in job_dirs:
            job_id = jdir.rstrip("/").split("/")[-1]
            try:
                meta = self.get_job(user_id, project_id, job_id)
                if status is None or meta.get("status") == status:
                    jobs.append(meta)
            except Exception:
                continue

        # Sort by submission time
        jobs.sort(key=lambda j: j.get("submitted_at", ""), reverse=True)
        return jobs

    def update_job_status(
        self,
        user_id: str,
        project_id: str,
        job_id: str,
        status: str,
        error: Optional[str] = None,
    ):
        """
        Update a job's status. Intended for the cluster worker to call.

        Parameters
        ----------
        status : str
            New status: "running", "completed", "failed".
        error : str, optional
            Error message if status is "failed".
        """
        meta = self.get_job(user_id, project_id, job_id)
        meta["status"] = status

        if status == "running":
            meta["started_at"] = _now_iso()
        elif status in ("completed", "failed"):
            meta["completed_at"] = _now_iso()

        if error:
            meta["error"] = error

        self._write_meta(
            self._job_meta_path(user_id, project_id, job_id), meta
        )
        print(f"✓ Job {job_id} status → {status}")

    # ==================================================================
    # RESULT UPLOAD / DOWNLOAD
    # ==================================================================

    def upload_result(
        self,
        user_id: str,
        project_id: str,
        job_id: str,
        data: np.ndarray,
        chunks: Optional[tuple[int, ...]] = None,
    ):
        """
        Upload model output (classification result) for a job.

        Typically called by the cluster worker after inference.

        Parameters
        ----------
        data : np.ndarray
            Classification result array (same spatial dims as region).
        chunks : tuple, optional
            Chunk sizes. Defaults to region_size (single chunk).
        """
        result_path = self._result_path(user_id, project_id, job_id)
        chunks = chunks or data.shape  # default: one chunk for small regions

        print(f"Uploading result: shape={data.shape}, dtype={data.dtype}")
        start = time.time()

        store = s3fs.S3Map(root=result_path, s3=self.s3, check=False)
        z = zarr.open(
            store,
            mode="w",
            shape=data.shape,
            chunks=chunks,
            dtype=data.dtype,
            zarr_format=2,
            compressor=self.compressor,
        )
        z[:] = data

        elapsed = time.time() - start
        print(f"✓ Uploaded result in {elapsed:.2f}s")

        # Mark job as completed
        self.update_job_status(user_id, project_id, job_id, "completed")

    def download_result(
        self,
        user_id: str,
        project_id: str,
        job_id: str,
    ) -> np.ndarray:
        """Download a job's classification result."""
        result_path = self._result_path(user_id, project_id, job_id)
        store = s3fs.S3Map(root=result_path, s3=self.s3, check=False)

        print(f"Downloading result for job {job_id}...")
        start = time.time()
        z = zarr.open(store, mode="r")
        data = z[:]
        elapsed = time.time() - start
        print(f"✓ Downloaded result {data.shape} in {elapsed:.2f}s")
        return data

    def open_result(
        self,
        user_id: str,
        project_id: str,
        job_id: str,
    ):
        """Open a result zarr lazily (for slicing without full download)."""
        result_path = self._result_path(user_id, project_id, job_id)
        store = s3fs.S3Map(root=result_path, s3=self.s3, check=False)
        return zarr.open(store, mode="r")

    def delete_job(self, user_id: str, project_id: str, job_id: str):
        """Delete a job and its result."""
        root = self._job_root(user_id, project_id, job_id)
        if self.s3.exists(root):
            self.s3.rm(root, recursive=True)
            print(f"✓ Deleted job {job_id}")
        else:
            print(f"⚠️  Job not found: {job_id}")


# ======================================================================
# DEMO / TEST
# ======================================================================

def demo():
    """
    Run a full demo: create project, upload volume, submit job,
    simulate model output, retrieve result.
    """
    bucket = "my-3d-data-test-bucket-12345"  # CHANGE THIS
    if len(sys.argv) > 1:
        bucket = sys.argv[1]

    mgr = S3DataManager(bucket)
    user_id = "demo_user"

    print("=" * 60)
    print("S3 Data Manager Demo")
    print("=" * 60)

    # --- 1. Create project ---
    project = mgr.create_project(
        user_id=user_id,
        name="Mouse Brain Sample 7",
        description="Test upload from demo script",
        voxel_size_nm=[4, 4, 30],
    )
    project_id = project["project_id"]

    # --- 2. Upload volume ---
    print("\nGenerating sample 3D volume (200x200x200)...")
    volume = np.random.randn(200, 200, 200).astype(np.float32)
    mgr.upload_volume(project, volume, chunks=(128, 128, 128))

    # --- 3. List projects ---
    print("\nProjects for user:")
    for p in mgr.list_projects(user_id):
        status = "✓ uploaded" if p["volume_uploaded"] else "○ no volume"
        print(f"  {p['project_id']} — {p['name']} ({status})")

    # --- 4. Submit a job (simulation auto-generates result) ---
    project = mgr.get_project(user_id, project_id)  # refresh metadata
    job = mgr.submit_job(project, region_origin=[50, 50, 50])
    job_id = job["job_id"]

    # --- 5. User retrieves result ---
    print("\n--- User retrieves result ---")
    job_info = mgr.get_job(user_id, project_id, job_id)
    print(f"Job status: {job_info['status']}")
    print(f"Region: origin={job_info['region_origin']}, size={job_info['region_size']}")

    result = mgr.download_result(user_id, project_id, job_id)
    print(f"Result shape: {result.shape}, dtype: {result.dtype}")
    print(f"Neuron voxels: {np.sum(result == 1)} / {result.size}")

    # --- 7. List jobs ---
    print("\nAll jobs for this project:")
    for j in mgr.list_jobs(user_id, project_id):
        print(f"  {j['job_id']} — {j['status']} — origin={j['region_origin']}")

    # --- 8. Cleanup ---
    print(f"\nClean up? (y/n): ", end="")
    if input().lower() == "y":
        mgr.delete_project(user_id, project_id)
    else:
        print(f"Data left at: s3://{bucket}/projects/{user_id}/{project_id}/")

    print("\n" + "=" * 60)
    print("✓ Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("S3 Data Manager")
        print("=" * 40)
        print()
        print("Usage:")
        print(f"  python {sys.argv[0]} <bucket-name>          Run demo")
        print()
        print("Example:")
        print(f"  python {sys.argv[0]} my-3d-data-test-bucket-12345")
        print()
        print("As a module:")
        print("  from s3_data_manager import S3DataManager")
        print('  mgr = S3DataManager("my-bucket")')
        print('  project = mgr.create_project("user_01", "My Scan")')
        print("  mgr.upload_volume(project, my_array)")
        print('  job = mgr.submit_job(project, region_origin=[100, 200, 50])')
    else:
        demo()