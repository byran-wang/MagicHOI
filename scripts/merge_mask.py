import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge object and hand masks into a single segmentation map.")
    parser.add_argument("--out_mask_path", required=True, help="Directory to save merged masks.")
    parser.add_argument("--object_mask_path", required=True, help="Directory containing object masks.")
    parser.add_argument("--hand_mask_path", required=True, help="Directory containing hand masks.")
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_segm_ids() -> dict:
    sys.path.append(str(project_root() / "code"))
    from src.utils.const import SEGM_IDS  # type: ignore

    return SEGM_IDS


def iter_mask_files(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.glob("*.png")):
        if path.is_file():
            yield path


def read_binary_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Unable to read mask: {path}")
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary


def merge_masks(
    object_mask_path: Path,
    hand_mask_path: Path,
    output_path: Path,
    segm_ids: dict,
) -> None:
    object_masks = list(iter_mask_files(object_mask_path))
    hand_masks = list(iter_mask_files(hand_mask_path))

    if len(object_masks) != len(hand_masks):
        raise ValueError("Number of object masks and hand masks do not match.")

    output_path.mkdir(parents=True, exist_ok=True)

    object_id = int(segm_ids["object"])
    hand_id = int(segm_ids["right"])

    for object_mask_file in tqdm(object_masks, desc="Merging masks"):
        hand_mask_file = hand_mask_path / object_mask_file.name

        if not hand_mask_file.exists():
            logging.warning("Missing hand mask for %s", object_mask_file.name)
            continue

        object_binary = read_binary_mask(object_mask_file)
        hand_binary = read_binary_mask(hand_mask_file)

        merged = np.zeros_like(object_binary, dtype=np.uint8)
        merged[object_binary > 0] = object_id
        merged[hand_binary > 0] = hand_id  # Hands override object where overlapping

        cv2.imwrite(str(output_path / object_mask_file.name), merged)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    segm_ids = load_segm_ids()

    out_mask_path = Path(args.out_mask_path).expanduser()
    object_mask_path = Path(args.object_mask_path).expanduser()
    hand_mask_path = Path(args.hand_mask_path).expanduser()

    logging.info("Merging masks into %s", out_mask_path)
    merge_masks(object_mask_path, hand_mask_path, out_mask_path, segm_ids)


if __name__ == "__main__":
    main()
