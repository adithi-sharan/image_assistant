#gets rid of dupicates(strict almost exact same) 
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
import imagehash


@dataclass(frozen=True)
class DedupeGroup:
    # Representative hash for the group and the filepaths in it
    phash: str
    paths: Tuple[str, ...]


def compute_phash(pil_img: Image.Image, hash_size: int = 16) -> imagehash.ImageHash:
    """
    Compute perceptual hash. hash_size=16 is more discriminative and strict than default 8.
    """
    return imagehash.phash(pil_img, hash_size=hash_size)


def group_strict_duplicates(
    items: List[dict],
    hash_size: int = 16,
    max_distance: int = 0,
) -> Tuple[Dict[str, int], List[DedupeGroup]]:
    """
    Groups items into duplicates using pHash Hamming distance.

    items: list of dicts with keys:
      - "path": str
      - "preview": PIL.Image.Image  (downsampled is fine)
      - (optional other keys)

    max_distance:
      - 0 = only exact same pHash (strictest)
      - 1-2 = still pretty strict for near-identical frames
    Returns:
      - path_to_group_index mapping
      - list of DedupeGroup
    """
    
    hashes: List[Tuple[str, imagehash.ImageHash]] = []
    for it in items:
        h = compute_phash(it["preview"], hash_size=hash_size)
        hashes.append((it["path"], h))

    # Greedy grouping: good enough for strict duplicates
    groups: List[List[str]] = []
    group_hashes: List[imagehash.ImageHash] = []

    for path, h in hashes:
        placed = False
        for gi, gh in enumerate(group_hashes):
            if (h - gh) <= max_distance:
                groups[gi].append(path)
                placed = True
                break
        if not placed:
            group_hashes.append(h)
            groups.append([path])

    path_to_group: Dict[str, int] = {}
    dedupe_groups: List[DedupeGroup] = []
    for i, paths in enumerate(groups):
        for p in paths:
            path_to_group[p] = i
        dedupe_groups.append(DedupeGroup(phash=str(group_hashes[i]), paths=tuple(paths)))

    return path_to_group, dedupe_groups
