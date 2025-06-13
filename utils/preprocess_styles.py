import argparse
import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    """Parse the arguments"""
    parser = argparse.ArgumentParser(description="Preprocess style images")
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Path to the unzipped images directory",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./data/style_images",
        help="Directory to save processed images",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="Size to resize images to"
    )
    return parser.parse_args()


def process_image(image_path: str, output_path: str, target_size: int) -> bool:
    """Process a single image by resizing and ensuring RGB format"""
    try:
        with Image.open(image_path) as img:
            # Skip grayscale images
            if img.mode in ["L", "LA"]:
                return False

            # Convert to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Direct resize to target size
            img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)

            # Save processed image
            img.save(output_path)
            return True
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return False


def main() -> None:
    args = parse_args()

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Get list of image files
    image_files = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    for ext in valid_extensions:
        image_files.extend(Path(args.data).glob(f"**/*{ext}"))

    print(f"Found {len(image_files)} images to process")

    # Process images
    successful = 0
    skipped_grayscale = 0

    for img_path in tqdm(image_files, desc="Processing images"):
        output_path = save_dir / f"{img_path.stem}.jpg"
        if process_image(img_path, output_path, args.size):
            successful += 1
        else:
            skipped_grayscale += 1

    print(f"Successfully processed {successful}/{len(image_files)} images")
    print(f"Skipped {skipped_grayscale} grayscale images")
    print(f"Processed images saved to {save_dir}")


if __name__ == "__main__":
    main()
