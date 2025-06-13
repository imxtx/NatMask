import cv2
import os
from tqdm import trange
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing the input images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the resized images.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for i in trange(30000):
        img = cv2.imread(os.path.join(args.input_dir, f"{i}.jpg"))
        img = cv2.resize(img, (256, 256))
        cv2.imwrite(os.path.join(args.output_dir, f"{i}.jpg"), img)
