"""
data_loader.py
===============

Utility functions for loading whole‑slide images and associated feature
embeddings.  This module provides helpers to parse a user‑provided
directory structure, load patch features and coordinates from Trident
format H5 files, and generate scaled coordinates appropriate for
display on thumbnails.

The expected directory layout beneath the user‑selected root is:

```
root_dir/
    slideA/
        slideA.svs           # the WSI (may be .svs, .tif, etc.)
        features/            # folder containing feature subdirectories
            model1/
                20x/
                    256px/
                        features.h5
                    512px/
                        features.h5
                40x/
                    256px/
                        ...
            model2/
                ...
    slideB/
        ...
```

Within each H5 file we expect at least two datasets: one for patch
embeddings (shape `(num_patches, feature_dim)`) and one for patch
coordinates (shape `(num_patches, 2)` or `(num_patches, 3)`).  The
coordinate dataset stores the top‑left corner of each patch in the
WSI’s level‑0 coordinate system【699992591352021†L149-L153】.  Trident also
records the patch size at level 0 in an attribute called
`patch_size_level0` or `patch_size_lv0`.  If this attribute is
missing, the loader attempts to infer it from the coordinate grid.

Functions in this module may raise runtime errors if required
dependencies (openslide, h5py) are missing.  The calling code should
handle these exceptions appropriately (e.g. by showing a message
dialog).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import h5py  # type: ignore
except ImportError:
    h5py = None  # type: ignore

try:
    import openslide  # type: ignore
except ImportError:
    openslide = None  # type: ignore

from PIL import Image


@dataclass
class SlideInfo:
    """Container describing a slide and its available feature sets.
    
    This dataclass stores metadata about a whole-slide image and all
    associated feature files that have been discovered for it. It provides
    a hierarchical mapping from model name → magnification → patch size →
    HDF5 file path.
    
    Attributes
    ----------
    name : str
        The slide name (typically the filename without extension, e.g.,
        "slideA" for "slideA.svs").
    image_path : str
        Full filesystem path to the whole-slide image file. Supported
        formats include .svs, .tif, .tiff, and .ndpi.
    models : Dict[str, Dict[str, Dict[str, str]]]
        Nested dictionary mapping model names to magnifications to patch
        sizes to HDF5 file paths. The structure is:
        ``models[model_name][magnification][patch_size] = h5_file_path``
        
        Example::
        
            {
                "resnet50": {
                    "20x": {
                        "256px": "/path/to/slideA.h5",
                        "512px": "/path/to/slideA_512.h5"
                    },
                    "40x": {
                        "256px": "/path/to/slideA_40x.h5"
                    }
                }
            }
        
        Model names are derived from directory names in the features folder,
        with any "features_" prefix removed automatically.
    """
    name: str
    image_path: str
    models: Dict[str, Dict[str, Dict[str, str]]]
    # models[model][magnification][tile_size] = h5_file_path


def _parse_trident_folder_name(folder: str) -> Optional[Tuple[str, str]]:
    """Extract magnification and patch size from a Trident-style folder name.

    Trident names follow the pattern ``<mag>_<patch>_...``, for example
    ``20x_512px_0px_overlap``.  Returns a ``(magnification, patch_size)``
    tuple such as ``("20x", "512px")``, or ``None`` if the name does not
    match the expected pattern.
    """
    m = re.match(r'^(\d+x)_(\d+px)', folder)
    if m:
        return m.group(1), m.group(2)
    return None


def _detect_format(root_dir: str) -> str:
    """Detect whether *root_dir* uses the legacy or Trident directory layout.

    Returns ``"legacy"`` when a ``features/`` subdirectory is present,
    ``"trident"`` when at least one subdirectory matches the Trident naming
    pattern (e.g. ``20x_512px_0px_overlap``), and ``"legacy"`` as a safe
    fallback otherwise.
    """
    if os.path.isdir(os.path.join(root_dir, 'features')):
        return 'legacy'
    for entry in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, entry)):
            if _parse_trident_folder_name(entry) is not None:
                return 'trident'
    return 'legacy'


def _parse_trident(root_dir: str, slide_files: Dict[str, str]) -> Dict[str, SlideInfo]:
    """Parse a Trident-format directory into a mapping of slide name → SlideInfo.

    Expected layout::

        root_dir/
            20x_512px_0px_overlap/
                features_virchow2/
                    slideA.h5
                features_gigapath/
                    slideA.h5

    The magnification and patch size are read from the top-level folder name;
    the model name is derived from the ``features_<model>`` subdirectory with
    the ``features_`` prefix stripped.
    """
    slides: Dict[str, SlideInfo] = {}

    for mag_patch_dir in sorted(os.listdir(root_dir)):
        mag_patch_path = os.path.join(root_dir, mag_patch_dir)
        if not os.path.isdir(mag_patch_path):
            continue
        result = _parse_trident_folder_name(mag_patch_dir)
        if result is None:
            continue
        mag, patch = result

        for model_dir in sorted(os.listdir(mag_patch_path)):
            model_path = os.path.join(mag_patch_path, model_dir)
            if not os.path.isdir(model_path):
                continue
            model_name = model_dir[9:] if model_dir.startswith('features_') else model_dir

            for slide_name, image_path in slide_files.items():
                h5_file = os.path.join(model_path, f"{slide_name}.h5")
                if os.path.isfile(h5_file):
                    if slide_name not in slides:
                        slides[slide_name] = SlideInfo(
                            name=slide_name,
                            image_path=image_path,
                            models={}
                        )
                    info = slides[slide_name]
                    if model_name not in info.models:
                        info.models[model_name] = {}
                    if mag not in info.models[model_name]:
                        info.models[model_name][mag] = {}
                    info.models[model_name][mag][patch] = h5_file
                    print(f"Found Trident features at: {h5_file}")

    return slides


def parse_root_directory(root_dir: str) -> Dict[str, SlideInfo]:
    """Parse the root directory to discover slides and feature files.

    Parameters
    ----------
    root_dir : str
        Path to the root directory containing .svs files and a features subdirectory.

    Returns
    -------
    Dict[str, SlideInfo]
        Mapping from slide name to a SlideInfo object describing the
        available feature files.
    """
    slides: Dict[str, SlideInfo] = {}
    if not os.path.isdir(root_dir):
        raise ValueError(f"Root directory does not exist: {root_dir}")

    # Find all slide files at the root level
    slide_files: Dict[str, str] = {}
    for fname in sorted(os.listdir(root_dir)):
        if fname.lower().endswith(('.svs', '.tif', '.tiff', '.ndpi')):
            slide_name = os.path.splitext(fname)[0]
            slide_files[slide_name] = os.path.join(root_dir, fname)

    if not slide_files:
        print(f"No slide files found in {root_dir}")
        return slides

    print(f"Found slide files: {list(slide_files.keys())}")

    fmt = _detect_format(root_dir)
    print(f"Detected directory format: {fmt}")

    if fmt == 'trident':
        return _parse_trident(root_dir, slide_files)

    # --- Legacy format: features/model/mag/patch/slide.h5 ---
    features_dir = os.path.join(root_dir, 'features')
    if not os.path.isdir(features_dir):
        print(f"No features directory found in {root_dir}")
        return slides

    for slide_name, image_path in slide_files.items():
        print(f"Looking for features for slide: {slide_name}")
        models: Dict[str, Dict[str, Dict[str, str]]] = {}

        for model_dir in sorted(os.listdir(features_dir)):
            model_path = os.path.join(features_dir, model_dir)
            if not os.path.isdir(model_path):
                continue

            model_name = model_dir[9:] if model_dir.startswith('features_') else model_dir
            models[model_name] = {}

            for mag_dir in sorted(os.listdir(model_path)):
                mag_path = os.path.join(model_path, mag_dir)
                if not os.path.isdir(mag_path):
                    continue

                models[model_name][mag_dir] = {}

                for patch_dir in sorted(os.listdir(mag_path)):
                    patch_path = os.path.join(mag_path, patch_dir)
                    if not os.path.isdir(patch_path):
                        continue

                    h5_file = os.path.join(patch_path, f"{slide_name}.h5")
                    if os.path.isfile(h5_file):
                        models[model_name][mag_dir][patch_dir] = h5_file
                        print(f"Found features at: {h5_file}")
                    else:
                        print(f"No features found for: {slide_name}")

        if models:
            slides[slide_name] = SlideInfo(
                name=slide_name,
                image_path=image_path,
                models=models
            )

    return slides


def load_features(h5_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load patch features and coordinates from a Trident H5 file.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file containing patch features and coordinates.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, int]
        A tuple `(features, coords, patch_size_lv0)` where
        `features` is an array of shape `(n_patches, feature_dim)`,
        `coords` is an array of shape `(n_patches, 2)` with the top‑left
        coordinates at level 0 magnification, and `patch_size_lv0` is the
        patch size at level 0 magnification (in pixels).

    Raises
    ------
    RuntimeError
        If h5py is not installed, or if the HDF5 file does not contain
        the required datasets (features and coordinates). The error message
        will indicate which dataset is missing.
    
    FileNotFoundError
        If the specified HDF5 file path does not exist.
    
    OSError
        If the HDF5 file cannot be opened (e.g., corrupted file, permission
        issues).
    
    Notes
    -----
    The function automatically detects feature and coordinate datasets by
    examining array shapes. If multiple datasets match the criteria, the
    first matching dataset is used. This heuristic approach makes the
    loader flexible but may fail if the HDF5 file has an unusual structure.
    
    If patch size cannot be determined from attributes, it is inferred by
    computing the median spacing between sorted, unique coordinate values.
    This assumes a regular grid layout of patches.
    """
    if h5py is None:
        raise RuntimeError("h5py is required to read feature files but is not installed.")
    with h5py.File(h5_path, 'r') as f:
        # Try standard dataset names first
        features_ds = None
        coords_ds = None
        # Search for dataset containing >10 features
        for name, dset in f.items():
            if isinstance(dset, h5py.Dataset):
                shape = dset.shape
                if len(shape) == 2 and shape[0] > 0:
                    if shape[1] > 10 and features_ds is None:
                        features_ds = dset
                    elif shape[1] in (2, 3) and coords_ds is None:
                        coords_ds = dset
        if features_ds is None or coords_ds is None:
            raise RuntimeError(f"Could not find features or coords dataset in {h5_path}")
        features = features_ds[...].astype(float)
        coords = coords_ds[...].astype(float)
        # Determine patch size at level 0
        patch_size_lv0 = None
        # Check file or dataset attributes
        for key in ['patch_size_level0', 'patch_size_lv0', 'patch_size']:
            if key in coords_ds.attrs:
                patch_size_lv0 = int(coords_ds.attrs[key])
                break
            if key in f.attrs:
                patch_size_lv0 = int(f.attrs[key])
                break
        if patch_size_lv0 is None:
            # Infer from coordinate grid spacing; assume regular grid
            xs = np.sort(np.unique(coords[:, 0]))
            ys = np.sort(np.unique(coords[:, 1]))
            # Compute median difference between sorted coordinates
            dxs = np.diff(xs)
            dys = np.diff(ys)
            # Filter zeros or near zeros
            dxs = dxs[dxs > 1e-3]
            dys = dys[dys > 1e-3]
            if dxs.size > 0:
                patch_size_lv0 = int(np.median(dxs))
            elif dys.size > 0:
                patch_size_lv0 = int(np.median(dys))
            else:
                patch_size_lv0 = 256  # fallback
        return features, coords[:, :2], patch_size_lv0


def load_thumbnail(image_path: str, max_size: int = 1024) -> Image.Image:
    """Load a downsampled thumbnail of a whole-slide image."""
    print(f"\nDebug: Starting load_thumbnail for {image_path}")
    print(f"Debug: Target max_size = {max_size}")
    
    # Use openslide if available for WSI formats
    if openslide is not None:
        try:
            print("Debug: Attempting OpenSlide path...")
            slide = openslide.OpenSlide(image_path)
            print(f"Debug: OpenSlide opened successfully. Dimensions: {slide.dimensions}")
            
            width, height = slide.dimensions
            best_level = slide.get_best_level_for_downsample(min(width, height) / max_size)
            print(f"Debug: Selected best_level = {best_level}")
            
            level_width, level_height = slide.level_dimensions[best_level]
            print(f"Debug: Level dimensions: {level_width} x {level_height}")
            
            scale = max(level_width, level_height) / float(max_size)
            thumb_w = int(level_width / scale)
            thumb_h = int(level_height / scale)
            print(f"Debug: Calculated thumbnail dimensions: {thumb_w} x {thumb_h}")
            
            thumb = slide.read_region((0, 0), best_level, slide.level_dimensions[best_level])
            thumb = thumb.convert('RGB')
            if thumb_w != level_width or thumb_h != level_height:
                print(f"Debug: Resizing from {level_width}x{level_height} to {thumb_w}x{thumb_h}")
                thumb = thumb.resize((thumb_w, thumb_h), Image.BILINEAR)
            print("Debug: OpenSlide path successful")
            return thumb
            
        except Exception as e:
            print(f"Debug: OpenSlide failed with error: {str(e)}")
            print("Debug: Falling back to PIL")
    else:
        print("Debug: OpenSlide not available")
    
    # Fallback: load via PIL with safe thumbnail generation
    print("Debug: Starting PIL path")
    try:
        # First, get image size without loading full image
        with Image.open(image_path) as img:
            print(f"Debug: Original image size: {img.size}")
            width, height = img.size
            # Calculate target size while maintaining aspect ratio
            scale = min(max_size / width, max_size / height)
            target_size = (int(width * scale), int(height * scale))
            print(f"Debug: Target size: {target_size}")
            
            # Process in chunks
            chunk_size = 1024
            print(f"Debug: Processing image in chunks of {chunk_size} rows")
            thumb = Image.new('RGB', target_size)
            
            for y in range(0, height, chunk_size):
                chunk_height = min(chunk_size, height - y)
                print(f"Debug: Processing chunk at y={y}, height={chunk_height}")
                chunk = img.crop((0, y, width, y + chunk_height))
                chunk.thumbnail((target_size[0], int(chunk_height * scale)), Image.BILINEAR)
                thumb.paste(chunk, (0, int(y * scale)))
            
            print("Debug: PIL processing completed successfully")
            return thumb
            
    except Exception as e:
        print(f"Debug: Error in PIL processing: {str(e)}")
        raise


def scale_coords_to_thumbnail(coords: np.ndarray, slide_dims: Tuple[int, int],
                              thumb_size: Tuple[int, int]) -> np.ndarray:
    """Scale level‑0 coordinates to thumbnail coordinates.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape `(n_patches, 2)` containing x,y coordinates in
        level‑0 space.
    slide_dims : Tuple[int, int]
        Dimensions `(width, height)` of the slide at level 0.
    thumb_size : Tuple[int, int]
        Dimensions `(width, height)` of the thumbnail image.

    Returns
    -------
    np.ndarray
        Scaled coordinates of shape `(n_patches, 2)`.

    Notes
    -----
    Level‑0 coordinates are scaled by the ratio of thumbnail to slide
    dimensions along each axis.  This assumes uniform scaling (no
    anisotropic mpp differences).  If the slide has different mpp in x
    and y, results may be slightly off.
    """
    slide_w, slide_h = slide_dims
    thumb_w, thumb_h = thumb_size
    sx = thumb_w / float(slide_w)
    sy = thumb_h / float(slide_h)
    scaled = np.zeros_like(coords, dtype=float)
    scaled[:, 0] = coords[:, 0] * sx
    scaled[:, 1] = coords[:, 1] * sy
    return scaled


def get_slide_dimensions(image_path: str) -> Optional[Tuple[int, int]]:
    """Get the full resolution dimensions of a whole-slide image.
    
    Parameters
    ----------
    image_path : str
        Path to the whole-slide image file.
        
    Returns
    -------
    Optional[Tuple[int, int]]
        Tuple of (width, height) in level-0 pixels, or None if dimensions
        cannot be determined.
        
    Notes
    -----
    Uses OpenSlide if available, otherwise falls back to PIL.
    """
    if openslide is not None:
        try:
            slide = openslide.OpenSlide(image_path)
            dims = slide.dimensions
            slide.close()
            return dims
        except Exception as e:
            print(f"Debug: OpenSlide failed to get dimensions: {e}")
    
    # Fallback to PIL
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception as e:
        print(f"Debug: PIL failed to get dimensions: {e}")
        return None


def is_openslide_available() -> bool:
    """Check if OpenSlide is available for adaptive zoom.
    
    Returns
    -------
    bool
        True if OpenSlide is installed and importable.
    """
    return openslide is not None