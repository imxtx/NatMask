import os
import numpy as np
from skimage.io import imread
from utils.helpers import read_landmark_106_array
from numpy.typing import NDArray


class CelebAHQ:
    def __init__(self, data_root: str, pair_list: str) -> None:
        super().__init__()
        self.pairs = []

        for pair in np.loadtxt(pair_list, dtype=str, delimiter=","):

            # read source info
            src_img = os.path.join(data_root, f"{pair[0]}.jpg")
            src_lms = os.path.join(data_root, f"{pair[0]}_landmark.txt")

            # read target info
            tar_img = os.path.join(data_root, f"{pair[1]}.jpg")

            self.pairs.append((src_img, src_lms, tar_img))

    def __getitem__(self, idx: int) -> tuple[NDArray, NDArray, NDArray]:
        src_image = imread(self.pairs[idx][0])

        # read facial landmarks of the source image
        with open(self.pairs[idx][1]) as f:
            kpts_str = f.readline().strip().split(" ")
            src_kpts = read_landmark_106_array(kpts_str)

        tar_image = imread(self.pairs[idx][2])

        return src_image, tar_image, src_kpts

    def __len__(self) -> int:
        return len(self.pairs)


class LFWPairs:
    def __init__(self, data_root: str, pair_list: str) -> None:
        super().__init__()
        self.pairs = []

        for p in np.loadtxt(pair_list, dtype=str, delimiter=","):
            name_src, img_src, name_tar, img_tar = p

            # read source info
            src_path = os.path.join(data_root, name_src)
            src_img = os.path.join(src_path, f"{name_src}_{img_src}.jpg")
            src_lms = os.path.join(src_path, f"{name_src}_{img_src}_landmark.txt")

            # read target info
            tar_img = os.path.join(data_root, name_tar, f"{name_tar}_{img_tar}.jpg")

            self.pairs.append((src_img, src_lms, tar_img))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[NDArray, NDArray, NDArray]:
        src_image = imread(self.pairs[idx][0])

        # read facial landmarks of the source image
        with open(self.pairs[idx][1]) as f:
            kpts_str = f.readline().strip().split(" ")
            src_kpts = read_landmark_106_array(kpts_str)

        tar_image = imread(self.pairs[idx][2])

        return src_image, tar_image, src_kpts


if __name__ == "__main__":
    # Test CelebAHQ dataset
    celeba_root = "/mnt/sdd/xietianxin/datasets/face/CelebAMask-HQ/CelebA-HQ-img-256"
    celeba_pairs = "./data/celeba_hq/dodging.txt"
    celeba_dataset = CelebAHQ(celeba_root, celeba_pairs)
    print(f"CelebAHQ dataset size: {len(celeba_dataset)}")

    # Test first sample
    src_img, tar_img, src_kpts = celeba_dataset[0]

    assert os.path.exists(
        celeba_dataset.pairs[0][0]
    ), f"Source image path {celeba_dataset.pairs[0][0]} does not exist"
    assert os.path.exists(
        celeba_dataset.pairs[0][1]
    ), f"Source landmarks path {celeba_dataset.pairs[0][1]} does not exist"
    assert os.path.exists(
        celeba_dataset.pairs[0][2]
    ), f"Target image path {celeba_dataset.pairs[0][2]} does not exist"

    print(f"The first sample:")
    print(f"Source image path: {celeba_dataset.pairs[0][0]}")
    print(f"Source landmarks path: {celeba_dataset.pairs[0][1]}")
    print(f"Target image path: {celeba_dataset.pairs[0][2]}")
    print(f"Source image shape: {src_img.shape}")
    print(f"Target image shape: {tar_img.shape}")
    print(f"Source landmarks shape: {src_kpts.shape}")

    # Test LFWPairs dataset
    lfw_root = "/mnt/sdd/xietianxin/datasets/face/LFW/lfw-deepfunneled"
    lfw_pairs = "./data/lfw/dodging.txt"
    lfw_dataset = LFWPairs(lfw_root, lfw_pairs)
    print(f"\nLFW dataset size: {len(lfw_dataset)}")

    # Test first sample
    src_img, tar_img, src_kpts = lfw_dataset[0]

    assert os.path.exists(
        lfw_dataset.pairs[0][0]
    ), f"Source image path {lfw_dataset.pairs[0][0]} does not exist"
    assert os.path.exists(
        lfw_dataset.pairs[0][1]
    ), f"Source landmarks path {lfw_dataset.pairs[0][1]} does not exist"
    assert os.path.exists(
        lfw_dataset.pairs[0][2]
    ), f"Target image path {lfw_dataset.pairs[0][2]} does not exist"

    print(f"The first sample:")
    print(f"Source image path: {lfw_dataset.pairs[0][0]}")
    print(f"Source landmarks path: {lfw_dataset.pairs[0][1]}")
    print(f"Target image path: {lfw_dataset.pairs[0][2]}")
    print(f"Source image shape: {src_img.shape}")
    print(f"Target image shape: {tar_img.shape}")
    print(f"Source landmarks shape: {src_kpts.shape}")
