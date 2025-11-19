#!/usr/bin/env python3
"""
Extract 8 axial slices through the largest tumor component (from the mask)
and save a 4-channel NumPy array per patient (T1, T1c, T2, FLAIR)
as a .npy file named after the patient.

Folder structure expected:
root/
 ├── PatientID_0001/
 │    ├── Timepoint_1/
 │    │    ├── (optionally one subfolder)/
 │    │    │    ├── .....brain_t1n.nii.gz
 │    │    │    ├── .....brain_t1c.nii.gz
 │    │    │    ├── .....brain_t2w.nii.gz
 │    │    │    ├── .....brain_t2f.nii.gz
 │    │    │    └── tumorMask.nii.gz
 │    ├── Timepoint_2/
 │    └── ...
"""

import argparse
import os
import re
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import label

# ------------------------------------------------------------
#                 Helper Functions
# ------------------------------------------------------------

def _find_timepoint_dir(patient_dir: Path) -> Path | None:
    """
    Find the correct subfolder for Timepoint_1.
    - Some datasets may have Timepoint_1 directly containing files,
      while others have an extra nested folder.
    - This function returns that correct folder.
    """
    tp1 = patient_dir / "Timepoint_1"
    if not tp1.exists():
        return None
    # If subdirectories exist under Timepoint_1, return the first one
    subdirs = sorted([p for p in tp1.iterdir() if p.is_dir()])
    return subdirs[0] if subdirs else tp1


def _glob_one(folder: Path, patterns: list[str]) -> Path | None:
    """
    Search for a file matching any of the provided regex patterns.
    - patterns: list of regex strings (e.g. ["t1c", "t1n"])
    - Returns the first matching file found, or None if not found.
    """
    pats = [re.compile(p, re.I) for p in patterns]  # case-insensitive
    for p in folder.iterdir():
        if p.is_file() and any(rx.search(p.name) for rx in pats):
            return p
    return None


def _load_nii(path: Path) -> np.ndarray:
    """
    Load a NIfTI (.nii or .nii.gz) file and return its voxel data as a NumPy array.
    """
    return np.asanyarray(nib.load(str(path)).get_fdata())


def _percentile_norm(vol: np.ndarray, p_low=1, p_high=99) -> np.ndarray:
    """
    Normalize an image volume to the [0,1] range using percentile clipping.
    - This reduces the impact of outlier intensities.
    - Percentiles (default 1–99) define the normalization window.
    """
    v = vol.astype(np.float32)
    lo, hi = np.percentile(v[np.isfinite(v)], [p_low, p_high])
    if hi <= lo:
        return np.zeros_like(v, dtype=np.float32)
    v = np.clip((v - lo) / (hi - lo), 0, 1)
    return v


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Identify and isolate the largest connected tumor region in the mask.
    - Binarizes the mask.
    - Labels connected components.
    - Keeps only the component with the most voxels.
    """
    m = (mask > 0).astype(np.int8)
    if m.sum() == 0:
        return m
    lbl, n = label(m)
    if n == 0:
        return m
    counts = np.bincount(lbl.ravel())
    counts[0] = 0  # ignore background
    k = counts.argmax()
    return (lbl == k).astype(np.uint8)


def _choose_axial_indices(mask_comp: np.ndarray, n_slices=8) -> list[int]:
    """
    Select 'n_slices' axial (z-axis) indices that cover the largest tumor region.
    - Finds all z-indices where the mask has non-zero voxels.
    - Evenly samples 8 indices between min and max tumor slices.
    """
    nz = np.where(mask_comp.any(axis=(0,1)))[0]
    if nz.size == 0:
        return []
    zmin, zmax = int(nz.min()), int(nz.max())
    if zmax == zmin:
        return [zmin] * n_slices
    idx = np.linspace(zmin, zmax, n_slices)
    rounded = np.clip(np.rint(idx).astype(int), zmin, zmax).tolist()
    # If range too narrow (few unique slices), spread evenly
    if len(set(rounded)) < n_slices:
        step = max((zmax - zmin) // max(n_slices - 1, 1), 1)
        rounded = list(range(zmin, zmin + step * n_slices, step))
        rounded = [min(zmax, i) for i in rounded]
    return rounded[:n_slices]

# ------------------------------------------------------------
#                 Main Patient Processing
# ------------------------------------------------------------

def process_patient(patient_dir: Path, dst_root: Path, n_slices=8) -> None:
    """
    Process one patient's folder:
    1. Find Timepoint_1 folder.
    2. Locate T1, T1c, T2, FLAIR, and mask files.
    3. Load and normalize scans.
    4. Extract 8 tumor slices (axial view).
    5. Stack modalities into (8, H, W, 4).
    6. Save as .npy file in a destination subfolder.
    """
    tp_dir = _find_timepoint_dir(patient_dir)
    if tp_dir is None:
        print(f"[skip] {patient_dir.name}: missing Timepoint_1")
        return

    # Try to locate each modality by regex pattern
    f_t1   = _glob_one(tp_dir, [r"t1n\.nii(\.gz)?$"])
    f_t1c  = _glob_one(tp_dir, [r"t1c\.nii(\.gz)?$"])
    f_t2   = _glob_one(tp_dir, [r"t2w\.nii(\.gz)?$"])
    f_flair= _glob_one(tp_dir, [r"t2f\.nii(\.gz)?$", r"flair\.nii(\.gz)?$"])
    f_mask = _glob_one(tp_dir, [r"tumormask\.nii(\.gz)?$", r"mask\.nii(\.gz)?$"])

    needed = {"T1": f_t1, "T1c": f_t1c, "T2": f_t2, "FLAIR": f_flair, "Mask": f_mask}
    missing = [k for k, v in needed.items() if v is None]
    if missing:
        print(f"[skip] {patient_dir.name}: missing {missing}")
        return

    # Load image volumes
    mask = _load_nii(f_mask)
    t1   = _load_nii(f_t1)
    t1c  = _load_nii(f_t1c)
    t2   = _load_nii(f_t2)
    flair= _load_nii(f_flair)

    # Ensure all modalities share same shape
    shapes = {tuple(arr.shape) for arr in [mask, t1, t1c, t2, flair]}
    if len(shapes) != 1:
        print(f"[skip] {patient_dir.name}: shapes differ {shapes}")
        return

    # Extract the largest tumor component
    comp = _largest_component(mask)

    # Determine which slices to take
    z_idx = _choose_axial_indices(comp, n_slices=n_slices)
    if not z_idx:
        print(f"[skip] {patient_dir.name}: empty tumor mask")
        return

    # Normalize all four modalities
    t1_n    = _percentile_norm(t1)
    t1c_n   = _percentile_norm(t1c)
    t2_n    = _percentile_norm(t2)
    flair_n = _percentile_norm(flair)

    # Combine 8 slices into a single 4-channel array
    # Shape: (8, H, W, 4)
    H, W, _ = t1_n.shape
    out = np.zeros((len(z_idx), H, W, 4), dtype=np.float32)
    for i, z in enumerate(z_idx):
        out[i, :, :, 0] = t1_n[:, :, z]
        out[i, :, :, 1] = t1c_n[:, :, z]
        out[i, :, :, 2] = t2_n[:, :, z]
        out[i, :, :, 3] = flair_n[:, :, z]

    # Save result in destination folder under patient ID
    dst_dir = dst_root / patient_dir.name
    dst_dir.mkdir(parents=True, exist_ok=True)
    np.save(dst_dir / f"{patient_dir.name}_tp1_slices.npy", out)
    print(f"[ok] {patient_dir.name}: saved {out.shape} -> {dst_dir}")

# ------------------------------------------------------------
#                 Command Line Interface
# ------------------------------------------------------------

def main():
    """
    Parse command-line arguments and process all patients under the source root.
    """
    ap = argparse.ArgumentParser(description="Extract tumor-centered slices and save stacked .npy per patient.")
    ap.add_argument("src_root", type=Path, help="Path to dataset root (contains patient ID folders).")
    ap.add_argument("dst_root", type=Path, help="Path to destination root where .npy files will be saved.")
    ap.add_argument("--slices", type=int, default=8, help="Number of axial slices to extract (default=8).")
    args = ap.parse_args()

    patients = [p for p in sorted(args.src_root.iterdir()) if p.is_dir()]
    if not patients:
        print("No patient folders found.")
        return

    for p in patients:
        process_patient(p, args.dst_root, n_slices=args.slices)

if __name__ == "__main__":
    main()
