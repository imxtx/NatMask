import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import torch.optim as optim
from torch import Tensor
from numpy.typing import NDArray
from omegaconf import DictConfig
from skimage.transform import estimate_transform, warp
from pytorch3d.renderer import MeshRenderer
from utils import (
    augment,
    clip_by_tensor,
    get_uv_textures_batch,
    render,
    write_obj_with_colors,
    sample_random_images,
    normalize_imagenet,
)
from models import PRNet
import logging

logging.basicConfig(level=logging.INFO)

import torchvision.models as models
from typing import Callable


class PRN:
    """Process of PRNet.
    Based on:
    https://github.com/YadiraF/PRNet/blob/master/api.py
    """

    def __init__(
        self,
        device: str,
        net: nn.Module,
        face_ind: NDArray,
        triangles: NDArray,
        triangles_tensor: Tensor,
    ) -> None:
        self.device = device
        self.resolution = 256
        self.MaxPos = self.resolution * 1.1
        self.face_ind = face_ind
        self.triangles = triangles
        self.triangles_tensor = triangles_tensor
        self.net = net

    def process(self, image: NDArray, image_info: NDArray) -> NDArray:
        if np.max(image_info.shape) > 4:  # Key points to get bounding box
            kpt = image_info
            if kpt.shape[0] > 3:
                kpt = kpt.T
            left = np.min(kpt[0, :])
            right = np.max(kpt[0, :])
            top = np.min(kpt[1, :])
            bottom = np.max(kpt[1, :])
        else:  # Bounding box
            # print("using bounding box")
            bbox = image_info
            left = bbox[0]
            top = bbox[1]
            right = bbox[2]
            bottom = bbox[3]
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        size = int(old_size * 1.6)
        # Crop image
        src_pts = np.array(
            [
                [center[0] - size / 2, center[1] - size / 2],
                [center[0] - size / 2, center[1] + size / 2],
                [center[0] + size / 2, center[1] - size / 2],
            ]
        )
        DST_PTS = np.array([[0, 0], [0, self.resolution - 1], [self.resolution - 1, 0]])
        tform = estimate_transform("similarity", src_pts, DST_PTS)
        image = image / 255.0
        cropped_image = warp(
            image, tform.inverse, output_shape=(self.resolution, self.resolution)
        )
        cropped_image = np.transpose(
            cropped_image[np.newaxis, :, :, :], (0, 3, 1, 2)
        ).astype(np.float32)
        cropped_image = torch.from_numpy(cropped_image)
        cropped_image = cropped_image.to(self.device)
        with torch.no_grad():
            cropped_pos = self.net(cropped_image)
        cropped_pos = cropped_pos.cpu().detach().numpy()
        cropped_pos = np.transpose(cropped_pos, (0, 2, 3, 1)).squeeze() * self.MaxPos
        # Restore
        cropped_vertices = np.reshape(cropped_pos, [-1, 3]).T
        z = cropped_vertices[2, :].copy() / tform.params[0, 0]
        cropped_vertices[2, :] = 1
        vertices = np.dot(np.linalg.inv(tform.params), cropped_vertices)
        vertices = np.vstack((vertices[:2, :], z))
        pos = np.reshape(vertices.T, [self.resolution, self.resolution, 3])
        return pos

    def process_batch(self, images: NDArray, kpts: NDArray) -> list[NDArray]:
        """Process a batch of N images with PRNet
        Args:
            images (NDArray): (N, H, W, 3) Face images
            kpts (NDArray): (N, 68, 2) Face key points
        Returns:
            list[NDArray]: (N, H, W, 3) Position maps
        """
        N, H, W, C = images.shape
        images = images / 255.0
        # Crop images
        cropped_images = []
        tforms = []
        for i in range(N):
            # Get bounding box
            kpt = kpts[i].T  # (68,2)->(2,68)
            left = np.min(kpt[0, :])
            right = np.max(kpt[0, :])
            top = np.min(kpt[1, :])
            bottom = np.max(kpt[1, :])
            old_size = (right - left + bottom - top) / 2
            center = np.array([(right + left) / 2.0, (bottom + top) / 2.0])
            size = int(old_size * 1.6)
            # Crop image
            src_pts = np.array(
                [
                    [center[0] - size / 2, center[1] - size / 2],
                    [center[0] - size / 2, center[1] + size / 2],
                    [center[0] + size / 2, center[1] - size / 2],
                ]
            )
            DST_PTS = np.array(
                [[0, 0], [0, self.resolution - 1], [self.resolution - 1, 0]]
            )
            tform = estimate_transform("similarity", src_pts, DST_PTS)
            tforms.append(tform)
            cropped_image = warp(
                images[i],
                tform.inverse,
                output_shape=(self.resolution, self.resolution),
            )
            cropped_image = np.transpose(cropped_image, (2, 0, 1)).astype(np.float32)
            cropped_images.append(cropped_image)

        # Process a batch of N images with PRNet
        cropped_images = np.stack(cropped_images)
        cropped_images = torch.from_numpy(cropped_images).to(self.device)
        with torch.no_grad():
            cropped_poses = self.net(cropped_images)
        cropped_poses = cropped_poses.detach().cpu().numpy()
        cropped_poses = np.transpose(cropped_poses, (0, 2, 3, 1)) * self.MaxPos

        # Restore
        poses = []
        for i in range(N):
            cropped_vertices = np.reshape(cropped_poses[i][None, :, :, :], [-1, 3]).T
            z = cropped_vertices[2, :].copy() / tforms[i].params[0, 0]
            cropped_vertices[2, :] = 1
            vertices = np.dot(np.linalg.inv(tforms[i].params), cropped_vertices)
            vertices = np.vstack((vertices[:2, :], z))
            pos = np.reshape(vertices.T, [self.resolution, self.resolution, 3])
            poses.append(pos)

        return np.stack(poses)

    def get_vertices(self, pos: NDArray) -> NDArray:
        """
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            vertices: the vertices(point cloud). shape = (num of points, 3). n is about 40K here.
        """
        all_vertices = np.reshape(pos, [self.resolution**2, -1])
        vertices = all_vertices[self.face_ind, :]
        return vertices

    def get_vertices_batch(self, poses: NDArray) -> NDArray:
        """
        Args:
            poses (NDArray): (N, 256, 256, 3) 3D position maps
        Returns:
            NDArray: (N, nVerts, 3) The vertices(point cloud), nVerts is about 40K.
        """
        vts = []
        for i in range(poses.shape[0]):
            all_vertices = np.reshape(poses[i], [self.resolution**2, -1])
            vertices = all_vertices[self.face_ind, :]
            vts.append(vertices)
        return np.stack(vts)

    def get_colors_from_texture(self, texture: Tensor) -> Tensor:
        all_colors = torch.reshape(texture, [self.resolution**2, -1])
        colors = all_colors[self.face_ind, :]
        return colors

    def generate_uv_coords(self) -> NDArray:
        resolution = self.resolution
        uv_coords = np.meshgrid(range(resolution), range(resolution))
        uv_coords = np.transpose(np.array(uv_coords), [1, 2, 0])
        uv_coords = np.reshape(uv_coords, [resolution**2, -1])
        uv_coords = uv_coords[self.face_ind, :]
        uv_coords = np.hstack((uv_coords[:, :2], np.zeros([uv_coords.shape[0], 1])))
        return (uv_coords / resolution).astype(np.float32)


def create_prnet(config: DictConfig) -> tuple[str, PRN, NDArray, NDArray, Tensor]:
    """Create PRNet model"""
    face_ind = np.loadtxt(config.prn.face_ind).astype(np.int32)
    triangles = np.loadtxt(config.prn.triangles).astype(np.int32)
    triangles_tensor = torch.tensor(triangles).to(config.device).long()

    net = PRNet(3, 3)

    logging.info("Loading pre-trained PRNet model ...")
    net.load_state_dict(torch.load(config.prn.prnet, map_location=config.device))
    net.to(config.device)

    return config.device, net, face_ind, triangles, triangles_tensor


class FaceMasker:
    """Face masker responsible for adding face mask"""

    def __init__(
        self,
        device: str,
        prn: PRN,
        renderer: MeshRenderer,
        face_masks: list[NDArray],
        is_aug: bool = True,
    ) -> None:
        """
        Args:
            prn (PRN): PRNet processor
            renderer (MeshRenderer): Mesh renderer
            face_masks (list[NDArray]): [Mask uv textures, Mask uv binary masks]
        """
        self.device = device
        self.prn = prn
        self.renderer = renderer
        assert len(face_masks) == 2
        self.mask_texture_templates, self.mask_bimask_templates = face_masks
        # set default attributes
        self.is_aug = is_aug

    def init_colors(self, line_colors: Tensor) -> None:
        """Initialize the line colors"""
        self.line_colors = line_colors.to(self.device)
        self.line_colors.requires_grad = True

    def init_textures(self, batch: NDArray, random: bool = False) -> Tensor:
        """Find best face mask for each face which leads to lowest/highest similarity
        Args:
            batch (NDArray): (2, N, H, W, C) Face images and binary masks with values
                                                between [0, 255]
            random (bool): Randomly choose a initial face mask if True
        Returns:
            Tensor: (N, H, W, C) Face mask uv textures
        """
        src_imgs, tar_imgs, kpts = batch

        nImg, hImg, wImg, cImg = src_imgs.shape
        nTex, hTex, wTex, cTex = self.mask_texture_templates.shape

        if random:
            best_idxs = np.random.randint(nTex, size=nImg)
        else:
            raise NotImplementedError(f"Best initial mask selection not implemented")

        # Set mask textures and corresponding binary masks
        mask_textures = self.mask_texture_templates[best_idxs]
        self.mask_textures = torch.tensor(mask_textures).to(self.device)
        self.mask_textures.requires_grad = True
        mask_bimasks = self.mask_bimask_templates[best_idxs]
        self.mask_bimasks = torch.tensor(mask_bimasks).to(self.device)

        return self.mask_textures

    def add_mask_batch(
        self, face_images: NDArray, face_kpts: NDArray, face_masks: list[NDArray] = None
    ) -> tuple[Tensor, Tensor]:
        """Add face mask to face.
        Args:
            face_images (NDArray): (N, H, W, C) Face images with values between [0, 255]
            face_kpts (NDArray): (N, 68, 2) Face landmarks
            face_masks (list[NDArray]): (2, N, H, W, C) [Mask uv textures, Mask uv binary masks]
        Returns:
            tuple[Tensor, Tensor]: (N, H, W, C) Face images wearing face masks,
            (N, H, W, C) Binary masks for face images wearing face masks
        """
        face_images, face_textures, verts, verts_uvs = self._get_assets(
            face_images, face_kpts
        )

        if face_masks is not None:  # True when initializing textures
            mask_textures, mask_bimasks = face_masks
            mask_textures = torch.tensor(mask_textures).to(self.device)
            mask_bimasks = torch.tensor(mask_bimasks).to(self.device)
        else:
            mask_textures, mask_bimasks = self.mask_textures, self.mask_bimasks

        # Generate new texture with face mask
        new_textures = self._get_masked_uv_texture(
            mask_textures, mask_bimasks, face_textures
        )

        self._textures_to_obj = new_textures.detach()
        self._verts_to_obj = verts.detach()

        # Render
        rendered_faces = render(
            self.renderer, new_textures, verts, verts_uvs, self.prn.triangles_tensor
        )
        rendered_faces = rendered_faces[:, :, :, :3]
        # imsave("./temp/rendered_face.jpg", rendered_faces[0].detach().cpu().numpy())
        rendered_masks = render(
            self.renderer,
            torch.ones_like(new_textures),
            verts,
            verts_uvs,
            self.prn.triangles_tensor,
        )
        rendered_masks = rendered_masks[:, :, :, :3]
        # imsave("./temp/rendered_mask.jpg", rendered_masks[0].detach().cpu().numpy())

        # Add masked face back to original image
        new_images = (
            face_images * (1 - rendered_masks) + rendered_faces * rendered_masks
        )
        # imsave("./temp/rendered_new_images.jpg", new_images[0].detach().cpu().numpy())

        # Post-process for face recognition
        new_images = clip_by_tensor(new_images, 0, 1)

        return new_images, rendered_masks

    def _get_masked_uv_texture(
        self, mask_textures: Tensor, mask_bimasks: Tensor, face_textures: Tensor
    ) -> Tensor:
        """Get face texture wearing face mask, augment if configured.
        Args:
            mask_textures (Tensor): (N, H, W, C) Face mask uv texture
            mask_bimasks (Tensor): (N, H, W, C) Binary mask for face mask texture
            face_textures (Tensor): (N, H, W, C) Face uv texture
        Returns:
            Tensor: (Augmented) face textures wearing face mask
        """
        # Augmentation
        if self.is_aug:
            mask_textures, mask_bimasks = augment(mask_textures, mask_bimasks)
        # Add face mask uv texture to face texture
        new_textures = (
            face_textures * (1 - mask_bimasks)  # face
            + mask_textures * mask_bimasks  # face mask
        )
        return new_textures

    def _get_assets(self, face_images: NDArray, face_kpts: NDArray) -> tuple:
        """Get resized images for prnet, textures, position maps, vertices, and
        uv coordinates.
        Args:
            face_images (NDArray): (N, H, W, C) Face images with values between [0, 255]
            face_kpts (NDArray): (N, 68, 2) Face landmarks
        Returns:
            tuple: Resized images for prnet, textures, position maps, vertices,
            and uv coordinates
        """
        assert face_images.ndim == 4

        N, H, W, C = face_images.shape
        if C == 4:
            face_images = face_images[:, :, :, :3]

        # resize to (256, 256) for PRNet
        temp = np.zeros((N, 256, 256, 3))
        for i in range(N):
            temp[i] = cv2.resize(face_images[i], (256, 256))
        face_images = temp
        face_kpts = 256 / H * face_kpts

        # 3d reconstruction -> get texture
        poses = self.prn.process_batch(face_images, face_kpts)
        # poses = norm_pos_map(poses)

        # get uv texture map from face image, (0,1)
        face_images = (face_images / 255.0).astype(np.float32)
        face_textures = get_uv_textures_batch(face_images, poses)

        # vertices generated by PRNet is inverted along y axis
        # (N, nVerts, 3)
        N, H, W, C = face_images.shape
        vertices = self.prn.get_vertices_batch(poses).astype(np.float32)
        vertices[:, :, 1] = H - 1 - vertices[:, :, 1]
        # uv coords generated by PRNet is inverted along y axis
        verts_uvs = self.prn.generate_uv_coords()[:, :2]
        verts_uvs[:, 1] = 1 - verts_uvs[:, 1]

        return (
            torch.tensor(face_images).to(self.device),
            torch.tensor(face_textures).to(self.device),
            torch.tensor(vertices).to(self.device),
            torch.tensor(verts_uvs).to(self.device),
        )

    def to(self, device: str = "cpu") -> None:
        """Move the model to the specified device"""
        self.device = device

    def save_obj_files(self, path_to_save: str) -> None:
        """Save obj files"""
        os.makedirs(path_to_save, exist_ok=True)
        N, _, _, _ = self._textures_to_obj.shape
        for i in range(N):
            colors = self.prn.get_colors_from_texture(self._textures_to_obj[i])
            filename = os.path.join(path_to_save, f"{i:4d}.obj")
            write_obj_with_colors(
                filename,
                self._verts_to_obj[i].cpu().numpy(),
                self.prn.triangles,
                colors.cpu().numpy(),
            )


class EvolutionaryStyleBank:
    """Evolutionary Style Bank"""

    def __init__(
        self,
        device: str,
        esb_size: int,
        K: float = 0.5,
        CR: float = 0.1,
    ) -> None:
        """Initialize the evolutionary style bank object"""
        if esb_size < 4:
            raise ValueError(
                "Style bank size is too small. At least 4 members are required."
            )
        self.esb_size = esb_size
        self.device = device
        self.K = K  # Differential weight
        self.CR = CR  # Crossover rate
        self._style_features = None
        self.lower_bounds = None
        self.upper_bounds = None

    def init_style_bank(self, initial_features: dict) -> None:
        """Initialize the style bank"""
        self._style_features = {
            layer: features.to(self.device)
            for layer, features in initial_features.items()
        }  # the shape of each item is (N, C, H, W) where N is esb_size

        # The population size is determined by the input features
        self.esb_size = list(self._style_features.values())[0].shape[0]

        # Compute the lower and upper bounds of the style features
        self.lower_bounds = {}
        self.upper_bounds = {}
        for layer, features in self._style_features.items():
            # Compute the minimum and maximum values of each feature dimension
            # along the population dimension (dim=0)
            self.lower_bounds[layer] = torch.min(features, dim=0)[0]
            self.upper_bounds[layer] = torch.max(features, dim=0)[0]

    def compute_ppg_loss(
        self,
        adv_imgs: Tensor,  # (esb_size, C, H, W)
        tar_imgs_list: list[Tensor],  # (n_surrogates, esb_size, C, H, W)
        surrogates: list[dict],
        id_criterion: Callable,
        single_esb: bool = False,
    ) -> float:
        """Compute the PPG loss for the given adv_imgs and tar_imgs_list on surrogate models.
        Args:
            adv_imgs (Tensor): (esb_size, C, H, W)
            tar_imgs_list (list[Tensor]): (n_surrogates, esb_size, C, H, W)
            surrogates (list[dict]): Surrogate models
            id_criterion (Callable): Identity criterion
        Returns:
            float: PPG loss
        """
        adv_feats_list = []
        tar_feats_list = []
        id_loss_list = []

        with torch.no_grad():
            for i in range(len(surrogates)):
                adv_imgs_i = surrogates[i]["transform"](
                    adv_imgs.permute(
                        0, 2, 3, 1
                    )  # (esb_size, C, H, W) -> (esb_size, H, W, C)
                )  # (esb_size, C, H, W)

                adv_feats_i = surrogates[i]["net"](adv_imgs_i)
                adv_feats_list.append(adv_feats_i)

                # Forward target examples
                tar_feats_i = surrogates[i]["net"](tar_imgs_list[i]).detach()
                tar_feats_list.append(tar_feats_i)
                id_loss_i = id_criterion(
                    adv_feats_i, tar_feats_i, single_esb=single_esb
                )
                id_loss_list.append(id_loss_i)  # (1,) or (esb_size, )

        if single_esb:
            return torch.stack(id_loss_list).mean(dim=0).tolist()  # (esb_size, )

        return torch.stack(id_loss_list).mean()  # (1,)

    def update_style_bank_adaptive(
        self,
        adv_imgs: Tensor,  # (batch_size, C, H, W)
        bi_masks: Tensor,  # (batch_size, C, H, W)
        tar_imgs_list: list[Tensor],  # (n_surrogates, batch_size, C, H, W)
        surrogates: list[dict],
        id_criterion: Callable,
        style_injector: "IdentityAwareStyleInjector",
        batch_size: int,
        single_esb: bool = True,
    ) -> None:
        """Update the style bank adaptively based on ESB size and batch size"""
        if self._style_features is None:
            raise ValueError(
                "Style bank is not initialized. Call init_style_bank() first."
            )

        # Case 1: esb_size < batch_size (ESB size = batch_size)
        # Each ESB item corresponds to one sample
        if self.esb_size < batch_size:
            # Create mapping: ESB item index -> sample index (1:1 mapping)
            esb_to_sample_mapping = list(range(batch_size))

            # Expand inputs to match ESB format
            expanded_adv_imgs = adv_imgs  # (batch_size, C, H, W)
            expanded_bi_masks = bi_masks  # (batch_size, C, H, W)
            expanded_tar_imgs_list = (
                tar_imgs_list  # (n_surrogates, batch_size, C, H, W)
            )

        # Case 2: esb_size >= batch_size
        # First batch_size items correspond to samples, remaining items are randomly assigned
        else:
            # Create mapping: ESB item index -> sample index
            esb_to_sample_mapping = list(range(batch_size))  # First batch_size items
            # Remaining ESB items randomly assigned to samples
            for i in range(batch_size, self.esb_size):
                sample_idx = np.random.randint(0, batch_size)
                esb_to_sample_mapping.append(sample_idx)

            # Expand inputs to match ESB format
            expanded_adv_imgs = adv_imgs[esb_to_sample_mapping]  # (esb_size, C, H, W)
            expanded_bi_masks = bi_masks[esb_to_sample_mapping]  # (esb_size, C, H, W)
            expanded_tar_imgs_list = [
                tar_img[esb_to_sample_mapping] for tar_img in tar_imgs_list
            ]  # (n_surrogates, esb_size, C, H, W)

        # Compute current PPG loss for all ESB items
        stylized_adv_imgs = style_injector.run_style_transfer(
            expanded_adv_imgs,
            expanded_bi_masks,
            style_features=self._style_features,
            tqdm_desc="Updating style bank: transferring current style features",
        )
        current_ppg_loss = self.compute_ppg_loss(
            stylized_adv_imgs,
            expanded_tar_imgs_list,
            surrogates,
            id_criterion,
            single_esb=single_esb,
        )

        # Run differential evolution for all ESB items simultaneously
        pop_size = self.esb_size

        # Generate all random indices for mutation
        all_indices = np.arange(pop_size)
        r1_indices = np.array(
            [
                np.random.choice(np.delete(all_indices, i), 3, replace=False)
                for i in range(pop_size)
            ]
        )

        # Pre-compute all trial vectors for all layers
        trial_vectors_dict = {}
        for layer, population in self._style_features.items():
            # Vectorized mutation for all individuals in this layer
            x_r1 = population[r1_indices[:, 0]]  # (pop_size, ...)
            x_r2 = population[r1_indices[:, 1]]  # (pop_size, ...)
            x_r3 = population[r1_indices[:, 2]]  # (pop_size, ...)

            # Create mutant vectors: v = x_r1 + K * (x_r2 - x_r3)
            mutant_vectors = x_r1 + self.K * (x_r2 - x_r3)
            # Clamp to bounds
            mutant_vectors = torch.clamp(
                mutant_vectors, self.lower_bounds[layer], self.upper_bounds[layer]
            )

            # Vectorized crossover for all individuals in this layer
            feature_shape = population[0].shape
            masks = torch.rand(pop_size, *feature_shape, device=self.device) < self.CR
            # Ensure at least one element from mutant is used for each individual
            for i in range(pop_size):
                j_rand = np.random.randint(0, masks[i].numel())
                masks[i].flatten()[j_rand] = True

            trial_vectors = population.clone()
            trial_vectors[masks] = mutant_vectors[masks]
            trial_vectors_dict[layer] = trial_vectors

        # Batch evaluate all candidates
        trial_stylized_adv_imgs = style_injector.run_style_transfer(
            expanded_adv_imgs,
            expanded_bi_masks,
            style_features=trial_vectors_dict,
            tqdm_desc="Updating style bank: transferring trial style features",
        )
        trial_ppg_losses = self.compute_ppg_loss(
            trial_stylized_adv_imgs,
            expanded_tar_imgs_list,
            surrogates,
            id_criterion,
            single_esb=single_esb,
        )

        # Selection: keep trial vectors if they're better
        for i in range(pop_size):
            if trial_ppg_losses[i] < current_ppg_loss[i]:
                for layer in self._style_features.keys():
                    self._style_features[layer][i] = trial_vectors_dict[layer][i]

    @property
    def style_features(self) -> dict:
        return self._style_features


class IdentityAwareStyleInjector:
    """Identity-Aware Style Injector"""

    def __init__(
        self,
        content_layers: list[str] = ["conv2_2"],
        style_layers: list[str] = [
            "conv1_1",
            "conv1_2",
            "conv2_1",
            "conv2_2",
            "conv3_1",
        ],
        device: str = "cpu",
        style_weight: float = 1000,
        n_iter: int = 300,
        esb_size: int = None,
        style_image_path: str = None,
        style_lr: float = 0.01,
    ) -> None:
        """Initialize the identity-aware style injector"""
        self.device = device
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.style_weight = style_weight
        self.n_iter = n_iter
        self.esb_size = esb_size
        self.style_image_path = style_image_path
        self.style_lr = style_lr

        self.features = {}
        self.hooks = []

        self.style_model = None

        self.bi_masks = None

    def _hook_fn(self, layer_name: str) -> Callable:
        """Create a hook function to capture the output of the specified layer."""

        def hook(model, input, output):
            self.features[layer_name] = output  # (N, C, H, W)

        return hook

    def prepare_style_model(self, model_name: str = "vgg19") -> None:
        """Add hooks to the model based on the specified content_layers and style_layers."""

        if model_name == "vgg19":
            cnn = (
                models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
                .features.to(self.device)
                .eval()
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Create a new nn.Sequential to store the modified, final model to be used
        new_model = nn.Sequential()
        # The index of the last hooked layer
        last_hooked_layer_index = -1

        i, j = 1, 1  # Increment j every time we see a conv

        # Use enumerate to get the index of each layer
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                name = f"conv{i}_{j}"
            elif isinstance(layer, nn.ReLU):
                name = f"relu{i}_{j}"
                layer = nn.ReLU(inplace=False)
                j += 1
            elif isinstance(layer, nn.MaxPool2d):
                name = f"pool_{i}"
                i += 1
                j = 1
            else:
                raise RuntimeError(f"Unrecognized layer: {layer.__class__.__name__}")

            new_model.add_module(name, layer)

            if name in self.content_layers or name in self.style_layers:
                current_layer = new_model[-1]
                handle = current_layer.register_forward_hook(self._hook_fn(name))
                self.hooks.append(handle)
                # Update the index of the last hooked layer
                last_hooked_layer_index = len(new_model) - 1

        # Truncate the model based on the last hooked layer
        if last_hooked_layer_index != -1:
            final_model = new_model[: last_hooked_layer_index + 1]
        else:
            final_model = new_model

        logging.info(
            f"Successfully registered {len(self.hooks)} hooks to the style model"
        )

        final_model.to(self.device)

        import copy

        self.style_model = copy.deepcopy(final_model)

        del cnn
        del new_model
        del final_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
        logging.info("Successfully removed all registered hooks")

    def _gram_matrix(self, input_tensor: Tensor) -> Tensor:
        """Compute Gram matrix."""
        b, c, h, w = input_tensor.size()
        features = input_tensor.reshape(b * c, h * w)
        G = torch.mm(features, features.t())
        return G.div(b * c * h * w)

    def _calculate_losses(
        self,
        generated_features: dict,
        content_features: dict,
        style_features: dict,
    ) -> tuple[Tensor, Tensor]:
        """Calculate content and style losses."""
        content_loss = 0
        style_loss = 0

        for layer in self.content_layers:
            content_loss += nn.functional.mse_loss(
                generated_features[layer], content_features[layer]
            )

        for layer in self.style_layers:
            G_generated = self._gram_matrix(generated_features[layer])
            G_style = self._gram_matrix(style_features[layer])
            try:
                style_loss += nn.functional.mse_loss(G_generated, G_style)
            except RuntimeError as e:
                print(f"Error in style loss calculation: {e}")
                print(f"Generated features shape: {generated_features[layer].shape}")
                print(f"Style features shape: {style_features[layer].shape}")
                raise e

        return content_loss, style_loss

    def init_esb(
        self, batch_size: int, de_K: float, de_CR: float, single_esb: bool = True
    ) -> None:
        """Initialize the style bank with random style images
        Args:
            batch_size: Number of samples in the batch
            de_K: Differential evolution parameter K
            de_CR: Differential evolution parameter CR
            single_esb: Whether to use single ESB (much more efficient with similar results)
        """

        if not single_esb:
            raise NotImplementedError(
                "Multi-ESB is very slow and not recommended. "
                "Please use single_esb=True (much more efficient with similar results)."
            )

        self.batch_size = batch_size

        # Determine actual ESB size based on the relationship between esb_size and batch_size
        if self.esb_size < batch_size:
            actual_esb_size = batch_size
            logging.info(
                f"ESB size ({self.esb_size}) < batch size ({batch_size}), using ESB size = {actual_esb_size}"
            )
        else:
            actual_esb_size = self.esb_size
            logging.info(
                f"ESB size ({self.esb_size}) >= batch size ({batch_size}), using ESB size = {actual_esb_size}"
            )

        # Create a single ESB with the determined size
        self.esb = EvolutionaryStyleBank(self.device, actual_esb_size, de_K, de_CR)

        # Sample random style images and initialize the ESB
        style_imgs = sample_random_images(self.style_image_path, actual_esb_size)
        style_imgs = style_imgs.to(self.device)
        style_features = self._get_style_features(style_imgs)
        self.esb.init_style_bank(style_features)

    def _get_style_features(self, style_imgs: Tensor) -> dict:
        """Get the style features of the style images"""
        self.style_model(normalize_imagenet(style_imgs))
        style_features = {
            layer: self.features[layer].detach() for layer in self.style_layers
        }  # the shape of each item is (N, C, H, W)
        self.features.clear()
        return style_features

    def _get_best_style_features(self) -> dict:
        """Get the best style features from the style bank"""
        if not hasattr(self, "esb") or self.esb is None:
            raise ValueError("Style bank is not initialized. Call init_esb() first.")

        # Return the style features from the single ESB
        # If esb_size < batch_size, each sample corresponds to one ESB item
        # If esb_size >= batch_size, each sample corresponds to the first batch_size ESB items
        style_features = self.esb.style_features

        # Extract features corresponding to batch_size samples
        if self.esb.esb_size < self.batch_size:
            # Each sample corresponds to one ESB item (ESB size = batch_size)
            return style_features
        else:
            # Each sample corresponds to the first batch_size ESB items
            batch_style_features = {}
            for layer, features in style_features.items():
                batch_style_features[layer] = features[: self.batch_size]
            return batch_style_features

    def update_esb(
        self,
        adv_imgs: Tensor,  # (N, C, H, W)
        bi_masks: Tensor,  # (N, C, H, W)
        tar_imgs_list: list[Tensor],  # (n_surrogates, N, C, H, W)
        surrogates: list[dict],  # (n_surrogates, )
        id_criterion: Callable,
        single_esb: bool = True,
    ) -> None:
        """Update the style bank"""
        assert len(tar_imgs_list) == len(surrogates)

        if not hasattr(self, "esb") or self.esb is None:
            raise ValueError("Style bank is not initialized. Call init_esb() first.")

        # Update the single ESB with new logic
        self.esb.update_style_bank_adaptive(
            adv_imgs,
            bi_masks,
            tar_imgs_list,
            surrogates,
            id_criterion,
            self,
            self.batch_size,
            single_esb,
        )

    def run_style_transfer(
        self,
        content_batch: Tensor,
        bi_masks: Tensor = None,
        style_features: dict = None,
        tqdm_desc: str = "Transferring best style features",
    ) -> Tensor:
        """Run style transfer with the given content_batch and style_features.
        Args:
            content_batch (Tensor): (N, C, H, W) Content images
            bi_masks (Tensor): (N, H, W, C) Binary masks for content images
            style_features (dict): Style features, if None, use the best style features
            tqdm_desc (str): Description for tqdm progress bar
        Returns:
            Tensor: (N, C, H, W) Style transferred images
        """
        content_batch = content_batch.to(self.device)

        assert content_batch.shape[1] == 3, "Content batch must have 3 channels (RGB)"

        bi_masks = bi_masks.to(self.device)

        # Extract content features
        self.style_model(normalize_imagenet(content_batch))
        content_features = {
            layer: self.features[layer].detach() for layer in self.content_layers
        }
        self.features.clear()

        if style_features is None:
            style_features = self._get_best_style_features()

        # Initialize the image to be optimized
        generated_image = content_batch.clone().contiguous().requires_grad_(True)

        optimizer = optim.Adam([generated_image], lr=self.style_lr)

        # Iterative optimization
        for step in tqdm.tqdm(
            range(self.n_iter), desc=tqdm_desc, position=1, leave=True
        ):
            optimizer.zero_grad()
            generated_image.data.clamp_(0, 1)

            # Hooks will be called here
            self.style_model(normalize_imagenet(generated_image))
            generated_features = {
                layer: self.features[layer]
                for layer in (self.content_layers + self.style_layers)
            }
            self.features.clear()

            content_loss, style_loss = self._calculate_losses(
                generated_features, content_features, style_features
            )
            total_loss = content_loss + self.style_weight * style_loss
            total_loss.backward()

            optimizer.step()

        generated_image.data.clamp_(0, 1)

        # Only transfer style to the masked region
        if self.bi_masks is not None:
            generated_image = (
                content_batch * (1 - bi_masks) + generated_image * bi_masks
            )

        return generated_image.detach()
