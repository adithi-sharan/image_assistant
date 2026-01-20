# scripts/scan_and_downsample.py
#purpose to scan pictures from the file and downsample them for preview
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Optional, Tuple
from PIL import Image, ImageOps
from exposure import exposure_score
from blur import blur_score_laplacian, normalize_blur
import csv 
from faces import face_count


VALID_EXTS = {".jpg", ".jpeg"}


@dataclass(frozen=True)
class ImageItem:
    path: Path
    original_size: Tuple[int, int]
    resized_size: Tuple[int, int]


def iter_jpegs(root: Path) -> Iterable[Path]:
    """Recursively yield JPEG/JPG files under root."""
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Input folder does not exist: {root}")

    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            yield p


def load_and_downsample(
    image_path: Path,
    long_edge: int = 768,
) -> Tuple[Image.Image, ImageItem]:
    """
    Load a JPEG and downsample so the long edge equals `long_edge`.
    Preserves aspect ratio. Applies EXIF orientation correctly.
    Returns (resized_image, metadata).
    """
    # Pillow opens lazily; use context manager to ensure file closes
    with Image.open(image_path) as im:
        # Fix orientation using EXIF (very common on phone/camera shots)
        im = ImageOps.exif_transpose(im)

        # Convert to RGB to avoid edge cases (e.g., CMYK JPEGs)
        if im.mode != "RGB":
            im = im.convert("RGB")

        orig_w, orig_h = im.size

        # Compute new size
        current_long = max(orig_w, orig_h)
        if current_long <= long_edge:
            # Already small enough, just return a copy
            resized = im.copy()
            meta = ImageItem(path=image_path, original_size=(orig_w, orig_h), resized_size=resized.size)
            return resized, meta

        scale = long_edge / float(current_long)
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))

        # High-quality downsample
        resized = im.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

        meta = ImageItem(path=image_path, original_size=(orig_w, orig_h), resized_size=(new_w, new_h))
        return resized, meta


def scan_folder_downsample(
    root: Path,
    long_edge: int = 768,
) -> Generator[Tuple[Image.Image, ImageItem], None, None]:
    """Yield (resized_image, metadata) for each JPEG found."""
    for path in iter_jpegs(root):
        try:
            img, meta = load_and_downsample(path, long_edge=long_edge)
            yield img, meta
        except Exception as e:
            # Skip unreadable/corrupt images but keep going
            print(f"[WARN] Failed to process {path}: {e}")


def save_preview(
    img: Image.Image,
    out_path: Path,
    quality: int = 85,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="JPEG", quality=quality, optimize=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scan a folder for JPEGs and downsample to previews.")
    parser.add_argument("--input", required=True, help="Input folder containing photos")
    parser.add_argument("--long-edge", type=int, default=768, help="Max long edge for preview")
    parser.add_argument("--save-previews", action="store_true", help="Save previews to disk")
    parser.add_argument("--preview-dir", default="outputs/previews", help="Where to save previews if enabled")
    parser.add_argument("--max", type=int, default=0, help="Process at most N images (0 = no limit)")
    parser.add_argument("--csv", default="outputs/results.csv", help="Where to write results CSV")
    parser.add_argument("--top", type=int, default=0, help="If > 0, only write top N rows to CSV")

    args = parser.parse_args()

    in_dir = Path(args.input)
    preview_dir = Path(args.preview_dir)

    count = 0
    rows = [] # for csv with results
    for img, meta in scan_folder_downsample(in_dir, long_edge=args.long_edge):
        count += 1

        # BLUR 
        raw_blur = blur_score_laplacian(img)
        blur = normalize_blur(raw_blur)
        # EXPOSURE
        exp, exp_tags = exposure_score(img)


        # TAGS  for images with issues
        tags = []
        if blur < 0.15:
            tags.append("BLURRY")
        tags.extend(exp_tags)
        #FACE COUNT
        faces = face_count(img)
        if faces > 0:
            tags.append(f"FACES_{faces}")


        # OVERALL SCORE
        overall = 0.55 * blur + 0.45 * exp
        print(
            f"{count:05d} {meta.path.name} "
            f"blur={blur:.2f} exp={exp:.2f} overall={overall:.2f} "
            f"{' '.join(tags)}"
        )
        
        rows.append({
            "path": str(meta.path),
            "filename": meta.path.name,
            "orig_w": meta.original_size[0],
            "orig_h": meta.original_size[1],
            "prev_w": meta.resized_size[0],
            "prev_h": meta.resized_size[1],
            "blur_raw": round(raw_blur, 2),
            "blur": round(blur, 4),
            "exp": round(exp, 4),
            "overall": round(overall, 4),
            "faces": faces,
            "tags": " ".join(tags),
          })

        #SAVE PREVIEW   doesnt always happen
        if args.save_previews:
            rel = meta.path.relative_to(in_dir)
            out_path = (preview_dir / rel).with_suffix(".jpg")
            save_preview(img, out_path)

        if args.max and count >= args.max:
            break
    print(f"Done. Processed {count} images.")
    
    # Sort highest overall first
    rows.sort(key=lambda r: r["overall"], reverse=True)
    out_csv = Path(args.csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "path", "filename", "orig_w", "orig_h", "prev_w", "prev_h",
        "blur_raw", "blur", "exp", "overall", "tags"
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote CSV: {out_csv} ({len(rows)} rows)")

