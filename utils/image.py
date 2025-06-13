import random
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms
from typing import Union
from torch import Tensor
from numpy.typing import NDArray
from skimage.io import imread, imsave
from .helpers import clip_by_tensor

_CORP_SIZE = (256, 256)


def _transform(image: Union[NDArray, Tensor]) -> Tensor:
    """
    Args:
        image (Union[NDArray,Tensor]): (H, W, C) or (N, H, W, C)
    Returns:
        Tensor: (N, C, H, W)
    """
    if isinstance(image, Tensor):
        if image.dim() == 3:
            image = torch.permute(image, (2, 0, 1))
        else:
            image = torch.permute(image, (0, 3, 1, 2))
    else:
        if image.ndim == 3:
            image = TF.to_tensor(image)
            image = image[None, :]
        else:
            img_list = []
            for i in range(image.shape[0]):
                img_list.append(TF.to_tensor(image[i]))
            image = torch.stack(img_list)
    return image


def transform_for_arcface(image: Union[NDArray, Tensor]) -> Tensor:
    image = _transform(image)
    image = TF.resize(image, [112, 112])
    image = TF.normalize(image, 0.5, 0.5)
    return image


def transform_for_facenet(image: NDArray) -> Tensor:
    image = _transform(image)
    image = TF.resize(image, [160, 160])
    image = TF.normalize(image, 0.5, 0.5)
    return image


def transform_for_cosface(image: NDArray) -> Tensor:
    image = _transform(image)
    image = TF.resize(image, [112, 112])
    image = TF.center_crop(image, output_size=[112, 96])
    image = TF.normalize(image, 0.5, 0.5)
    return image


def resize_batch(images: NDArray, size: tuple[int, int]) -> NDArray:
    """Resize a batch of images"""
    assert images.ndim == 4
    assert len(size) == 2
    N, H, W, C = images.shape
    temp = np.zeros((N, *size, C))
    for i in range(N):
        temp[i] = cv2.resize(images[i], size)
    return temp


def split_image(image: NDArray, dim=1) -> tuple[NDArray, NDArray]:
    """Split a long image (H, W) into two images (H, W/2) or (H/2, W)"""
    assert dim in [0, 1], "Dim must be 0 or 1 for HWC images"
    assert len(image.shape) == 4, "image must be a 4D(N,H,W,C) array"
    N, H, W, C = image.shape
    if dim == 1:
        return image[:, :, : int(W / 2), :], image[:, :, int(W / 2) :, :]
    else:
        return image[:, : int(H / 2), :, :], image[:, int(H / 2) :, :, :]


def denormalize(image: Tensor) -> Tensor:
    """Denormalize the image"""
    """[-1,1] -> [0,1]"""
    return image * 0.5 + 0.5


def augment(mask_texture: Tensor, mask_bimask: Tensor) -> tuple[Tensor, Tensor]:
    """Augmentation for mask uv texture
    Args:
        mask_texture (tensor): (N, H, W, C) UV textures of face masks
        mask_bimask (tensor): (N, H, W, C) Corresponding binary masks
    Returns:
        tuple[Tensor, Tensor]: (N, H, W, C)
    """
    # (n,h,w,c) -> (n,c,h,w)
    mask_texture = torch.permute(mask_texture, (0, 3, 1, 2))
    mask_bimask = torch.permute(mask_bimask, (0, 3, 1, 2))

    # random flip
    if np.random.rand() > 0.5:
        mask_texture = torch.flip(mask_texture, dims=(3,))
        mask_bimask = torch.flip(mask_bimask, dims=(3,))
    # random scale
    if np.random.rand() > 0.5:
        sf = _random_factor()
        mask_texture = TF.center_crop(
            F.interpolate(mask_texture, scale_factor=sf, mode="bilinear"),
            _CORP_SIZE,
        )
        mask_bimask = TF.center_crop(
            F.interpolate(
                mask_bimask,
                scale_factor=sf,
                mode="bilinear",
            ),
            _CORP_SIZE,
        )
    # random brightness
    if np.random.rand() > 0.5:
        bf = _random_factor()
        mask_texture = TF.adjust_brightness(mask_texture, bf)
    # random contrast
    if np.random.rand() > 0.5:
        cf = _random_factor()
        mask_texture = TF.adjust_contrast(mask_texture, cf)

    mask_texture = clip_by_tensor(mask_texture, 0, 1)

    # (n,c,h,w) -> (n,h,w,c)
    mask_texture = torch.permute(mask_texture, (0, 2, 3, 1))
    mask_bimask = torch.permute(mask_bimask, (0, 2, 3, 1))

    return mask_texture, mask_bimask


def load_mask_textures(texture_dir: str, uv_face_mask: str) -> tuple[NDArray, NDArray]:
    """
    Args:
        texture_dir (str): Path to texture directory
        uv_face_mask (str): Path to uv face mask image
    Returns:
        tuple[NDArray, NDArray]: mask textures and corresponding binary masks
    """
    uv_face = imread(uv_face_mask, as_gray=True)
    uv_face = (uv_face / 255.0).astype(np.float32)
    filelist = [
        os.path.join(texture_dir, filename) for filename in os.listdir(texture_dir)
    ]

    mask_textures, mask_bimasks = [], []

    for file in filelist:
        mask_texture = imread(file)
        mask_texture = (mask_texture / 255.0).astype(np.float32)
        # face mask texture
        assert mask_texture.shape[2] == 4, "must have 4 channels"
        mask_textures.append(mask_texture[:, :, :3])
        # extract alpha channel as binary mask
        mask_bimask = mask_texture[:, :, 3] * uv_face
        mask_bimasks.append(mask_bimask[:, :, np.newaxis])

    return np.stack(mask_textures), np.stack(mask_bimasks)


def _random_factor() -> float:
    """Random factor for augmentation"""
    factor = np.random.randint(21) * 0.01 + 0.90  # [0.90,1.10]
    return factor


def save_images(save_path: str, images: NDArray) -> None:
    """Save images to a directory"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if np.max(images) <= 1.0:
        images = images * 255

    for i in range(images.shape[0]):
        name = f"{i:04d}"
        fname = os.path.join(save_path, f"{name}.png")
        imsave(fname, images[i].astype(np.uint8))


def sample_random_images(img_dir: str, n_samples: int) -> Tensor:
    """Randomly sample N images from a directory and return as a tensor.
    Args:
        img_dir: Path to image directory
        n_samples: Number of images to sample
    Returns:
        Tensor of shape (N,C,H,W) containing sampled images
    """
    # Get all image files
    img_files = [
        f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png", ".jpeg"))
    ]
    if len(img_files) < n_samples:
        raise ValueError(
            f"Directory contains only {len(img_files)} images, but {n_samples} requested"
        )

    # Randomly sample N files
    selected_files = random.sample(img_files, n_samples)

    # Load and preprocess images
    images = []
    for f in selected_files:
        img_path = os.path.join(img_dir, f)
        img = Image.open(img_path).convert("RGB")
        img = transforms.ToTensor()(img)  # Convert to tensor (C,H,W), [0,1]
        images.append(img)

    # Stack into single tensor
    return torch.stack(images)  # Shape: (N,C,H,W)


def normalize_imagenet(x: Tensor) -> Tensor:
    """Normalize input tensor with ImageNet statistics.
    Args:
        x: Input tensor of shape (N,C,H,W)
    Returns:
        Normalized tensor of same shape
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
    return (x - mean) / std
