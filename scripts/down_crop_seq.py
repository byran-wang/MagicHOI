import argparse
import logging
from pathlib import Path
from typing import Iterable, Sequence, Tuple

from PIL import Image
from tqdm import tqdm


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Downscale and crop image sequences.")
    parser.add_argument("--source_dir", required=True, help="Directory containing original sequences.")
    parser.add_argument("--target_dir", required=True, help="Directory to write processed data.")
    parser.add_argument(
        "--sub_folders",
        type=lambda s: [item for item in s.split(",") if item],
        required=True,
        help="Comma separated list of subfolders to process (e.g. images,masks).",
    )
    parser.add_argument("--scaler", type=float, default=1.0, help="Factor used to downscale images.")
    parser.add_argument(
        "--crop_size",
        type=lambda s: tuple(int(item) for item in s.split(",")),
        default=(640, 480),
        help="Desired crop size expressed as width,height.",
    )
    return parser.parse_args()


def iter_image_files(folder: Path) -> Iterable[Path]:
    for item in sorted(folder.iterdir()):
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES:
            yield item


def downscale_crop_images(
    source_dir: Path,
    target_dir: Path,
    sub_folders: Sequence[str],
    scaler: float,
    crop_size: Tuple[int, int],
) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"Input folder '{source_dir}' does not exist")

    target_dir.mkdir(parents=True, exist_ok=True)
    crop_width, crop_height = crop_size

    for sub_folder in sub_folders:
        src_sub = source_dir / sub_folder
        if not src_sub.exists():
            logging.warning("Skip missing sub-folder: %s", src_sub)
            continue

        dst_sub = target_dir / sub_folder
        dst_sub.mkdir(parents=True, exist_ok=True)

        for image_path in tqdm(iter_image_files(src_sub), desc=f"Scaling and cropping {sub_folder}"):
            output_path = dst_sub / f"{image_path.stem}.png"
            try:
                with Image.open(image_path) as img:
                    new_size = (max(1, int(round(img.width * scaler))), max(1, int(round(img.height * scaler))))
                    downscaled = img.resize(new_size, Image.Resampling.LANCZOS)

                    if new_size[0] < crop_width or new_size[1] < crop_height:
                        logging.warning(
                            "Skipping %s because scaled size %s is smaller than crop size %s",
                            image_path.name,
                            new_size,
                            crop_size,
                        )
                        continue

                    crop_left = new_size[0] // 2 - crop_width // 2
                    crop_top = new_size[1] // 2 - crop_height // 2
                    cropped = downscaled.crop(
                        (crop_left, crop_top, crop_left + crop_width, crop_top + crop_height)
                    )
                    cropped.save(output_path)
            except Exception as exc:  # pylint: disable=broad-except
                logging.error("Error processing '%s': %s", image_path, exc)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    source_dir = Path(args.source_dir).expanduser()
    target_dir = Path(args.target_dir).expanduser()
    crop_size = tuple(args.crop_size)  # Ensure tuple even if parser returns list/tuple

    logging.info("Scaling and cropping images to: %s", target_dir)
    downscale_crop_images(
        source_dir=source_dir,
        target_dir=target_dir,
        sub_folders=args.sub_folders,
        scaler=args.scaler,
        crop_size=crop_size,
    )


if __name__ == "__main__":
    main()
