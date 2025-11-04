import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_segm_ids() -> dict:
    sys.path.append(str(project_root() / "code"))
    from src.utils.const import SEGM_IDS  # type: ignore

    return SEGM_IDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate masked RGB/RGBA images.")
    parser.add_argument("--seq_name", required=True, help="Sequence name to process.")
    parser.add_argument("--source_dir", default="./data", help="Directory containing images and masks subfolders.")
    parser.add_argument("--target_dir", default="./data", help="Directory where processed images are written.")
    parser.add_argument(
        "--process_type",
        choices=["object", "hand"],
        default="object",
        help="Select which segmentation id to retain.",
    )
    parser.add_argument(
        "--rgba_format",
        action="store_true",
        help="Write output images with alpha channel derived from the mask.",
    )
    return parser.parse_args()


def iter_image_pairs(rgb_dir: Path, mask_dir: Path) -> Iterable[Tuple[Path, Path]]:
    rgb_paths = sorted(rgb_dir.glob("*.png"))
    mask_paths = sorted(mask_dir.glob("*.png"))

    if len(rgb_paths) != len(mask_paths):
        raise ValueError("RGB image count does not match mask count.")

    for rgb_path, mask_path in zip(rgb_paths, mask_paths):
        yield rgb_path, mask_path


def build_condition(mask_np: np.ndarray, segm_ids: dict, process_type: str) -> np.ndarray:
    if mask_np.ndim == 3:
        mask_np = mask_np[:, :, 0]

    if process_type == "object":
        return mask_np == segm_ids["object"]
    if process_type == "hand":
        return (mask_np == segm_ids["right"]) | (mask_np == segm_ids.get("left", -1))

    raise ValueError(f"Unsupported process_type '{process_type}'")


def process_images(
    rgb_dir: Path,
    mask_dir: Path,
    target_dir: Path,
    segm_ids: dict,
    process_type: str,
    rgba_format: bool,
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)

    for rgb_path, mask_path in tqdm(list(iter_image_pairs(rgb_dir, mask_dir)), desc="Generating masks"):
        with Image.open(rgb_path) as rgb_img, Image.open(mask_path) as mask_img:
            rgb_np = np.array(rgb_img.convert("RGB"))
            mask_np = np.array(mask_img)

        condition = build_condition(mask_np, segm_ids, process_type)

        alpha_channel = np.zeros_like(condition, dtype=np.uint8)
        alpha_channel[condition] = 255

        rgb_np[~condition] = 255

        if rgba_format:
            output_np = np.concatenate([rgb_np, alpha_channel[:, :, None]], axis=2)
        else:
            output_np = rgb_np

        output_path = target_dir / rgb_path.name
        Image.fromarray(output_np).save(output_path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    segm_ids = load_segm_ids()

    source_dir = Path(args.source_dir).expanduser()
    target_dir = Path(args.target_dir).expanduser()

    rgb_dir = source_dir / "images"
    mask_dir = source_dir / "masks"
    if not rgb_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError("Source directory must contain 'images' and 'masks' subfolders.")

    logging.info("Generating %s masks to %s", args.process_type, target_dir)
    process_images(
        rgb_dir=rgb_dir,
        mask_dir=mask_dir,
        target_dir=target_dir,
        segm_ids=segm_ids,
        process_type=args.process_type,
        rgba_format=args.rgba_format,
    )


if __name__ == "__main__":
    main()
