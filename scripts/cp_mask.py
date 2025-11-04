import argparse
import logging
import shutil
from pathlib import Path
from typing import Iterable

from PIL import Image
from tqdm import tqdm


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy masks while normalizing names and format.")
    parser.add_argument("--source_dir", required=True, help="Directory with source mask images.")
    parser.add_argument("--target_dir", required=True, help="Directory where processed masks are written.")
    return parser.parse_args()


def iter_image_files(source_dir: Path) -> Iterable[Path]:
    for path in sorted(source_dir.iterdir()):
        if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
            yield path


def copy_masks(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(iter_image_files(source_dir), desc="Copying masks"):
        try:
            index = int(image_path.stem)
        except ValueError:
            logging.warning("Skip file without numeric stem: %s", image_path.name)
            continue

        target_path = target_dir / f"{index:04d}.png"

        if image_path.suffix.lower() == ".png":
            shutil.copyfile(image_path, target_path)
        else:
            with Image.open(image_path) as img:
                img.convert("RGB").save(target_path, "PNG")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    source_dir = Path(args.source_dir).expanduser()
    target_dir = Path(args.target_dir).expanduser()

    logging.info("Copying masks from %s to %s", source_dir, target_dir)
    copy_masks(source_dir, target_dir)


if __name__ == "__main__":
    main()
