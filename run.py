import os
import hydra
import logging
import wandb
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from dataset import CelebAHQ, LFWPairs
from models import FACE_MODELS
from natmask import FaceMasker, PRN, create_prnet, IdentityAwareStyleInjector
from utils.render import create_renderer
from utils.image import (
    load_mask_textures,
    save_images,
    resize_batch,
)
from utils.optimizer import NatMaskOptimizer
from utils.loss import dodging_loss, impersonation_loss, tv_loss
from datetime import datetime
from torchvision.utils import make_grid
from torch import Tensor
from typing import Callable


def run_one_batch(
    masker: FaceMasker,
    style_injector: IdentityAwareStyleInjector,
    batch_data: tuple[Tensor, Tensor, Tensor],
    batch_id: int,
    n_batches: int,
    surrogates: list[dict],
    id_criterion: Callable,
    cfg: DictConfig,
) -> None:
    """Run one batch of the attack"""
    # Initialize face masks
    src_imgs, tar_imgs, kpts = batch_data
    N, C, H, W = src_imgs.shape
    src_imgs_np, tar_imgs_np, kpts_np = (
        src_imgs.numpy(),  # n,h,w,c
        tar_imgs.numpy(),  # n,h,w,c
        kpts.numpy(),
    )

    mask_textures = masker.init_textures(
        [src_imgs_np, tar_imgs_np, kpts_np], random=True
    )

    style_injector.init_esb(N, cfg.de_K, cfg.de_CR, single_esb=cfg.single_esb)

    # Create new optimizer for each batch
    optimizer = NatMaskOptimizer([mask_textures], lr=cfg.lr, momentum=cfg.momentum)

    tar_imgs_list = [
        surrogate["transform"](tar_imgs_np).to(cfg.device) for surrogate in surrogates
    ]  # n,c,h,w

    tqdm_desc = f"Device {cfg.device}, Batch {batch_id:02d}/{n_batches}, Batch Size {N}"
    for step in trange(cfg.n_iter, desc=tqdm_desc, position=0, leave=True):
        # Get masked images, (0,1), (n,h,w,c), (256x256)
        adv_imgs, bi_masks = masker.add_mask_batch(src_imgs_np, kpts_np)
        adv_np = adv_imgs.detach().clone().cpu().numpy()  # n,h,w,c

        # Inject style with best style features
        stylized_adv_imgs = style_injector.run_style_transfer(
            adv_imgs.detach().permute(0, 3, 1, 2),  # n,h,w,c -> n,c,h,w
            bi_masks.permute(0, 3, 1, 2),  # n,h,w,c -> n,c,h,w
        ).permute(
            0, 2, 3, 1
        )  # n,h,w,c -> n,c,h,w
        iasi_loss = F.mse_loss(stylized_adv_imgs, adv_imgs)

        # Transforms and forward pass
        adv_feats_list = []
        tar_feats_list = []
        id_loss_list = []
        tv_loss_list = []
        sims_list = []

        for i in range(len(surrogates)):
            # Transform and forward adv examples
            adv_imgs_i = surrogates[i]["transform"](adv_imgs)  # n,c,h,w
            adv_feats_i = surrogates[i]["net"](adv_imgs_i)
            adv_feats_list.append(adv_feats_i)

            # Forward target examples
            tar_feats_i = surrogates[i]["net"](tar_imgs_list[i]).detach()
            tar_feats_list.append(tar_feats_i)

            # Calculate losses
            id_loss_i = id_criterion(adv_feats_i, tar_feats_i)
            id_loss_list.append(id_loss_i)

            tv_loss_i = tv_loss(adv_imgs_i, cfg.tv_weight)
            tv_loss_list.append(tv_loss_i)

            sims_i = F.cosine_similarity(adv_feats_i, tar_feats_i)
            sims_list.append(sims_i)

        # Average losses
        id_loss_all = sum(id_loss_list) / len(surrogates)
        tv_loss_all = sum(tv_loss_list) / len(surrogates)
        sims = sum(sims_list) / len(surrogates)

        loss = id_loss_all + tv_loss_all + cfg.iasi_weight * iasi_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update style bank and best style features
        style_injector.update_esb(
            adv_imgs.detach().permute(0, 3, 1, 2),  # n,c,h,w
            bi_masks.permute(0, 3, 1, 2),  # n,c,h,w
            tar_imgs_list,  # (n_surrogates, N, C, H, W)
            surrogates,  # (n_surrogates, )
            id_criterion,
            single_esb=cfg.single_esb,
        )

        # Log to wandb
        wandb.log(
            {
                f"loss/batch_{batch_id:02d}": loss.item(),
                f"similarity/batch_{batch_id:02d}": sims.mean().item(),
                "run_step": step,
            },
        )
        # Log images
        if step % cfg.wandb.n_log_interval == 0 or step == cfg.n_iter - 1:
            imgs = np.concatenate(
                [
                    resize_batch(src_imgs_np, (256, 256)) / 255.0,  # n,h,w,c
                    resize_batch(tar_imgs_np, (256, 256)) / 255.0,  # n,h,w,c
                    adv_np,  # n,h,w,c
                ],
                axis=2,
            )  # n,h,w*3,c
            imgs_tensor = torch.from_numpy(imgs).permute(0, 3, 1, 2)  # n,c,h,w
            imgs_grid = make_grid(
                imgs_tensor,
                nrow=8,
            )

            # Log each grid separately
            wandb.log(
                {
                    f"images/batch_{batch_id:02d}": wandb.Image(
                        imgs_grid, caption=f"Step {step}"
                    ),
                    "run_step": step,
                },
            )

    # Save adversarial images
    if cfg.save_images:
        save_path = f"{cfg.results_dir}/{cfg.test_dataset}/{cfg.target.model}/{cfg.attack}/batch_{batch_id:02d}/images"
        os.makedirs(save_path, exist_ok=True)
        # Concatenate images along width dimension
        tar_256 = resize_batch(tar_imgs_np, (256, 256)) / 255.0
        pair_imgs = np.concatenate([adv_np, tar_256], axis=2)
        save_images(save_path, pair_imgs)
        logging.info(f"Successfully saved adversarial images to {save_path}")

    # Save obj files
    if cfg.save_obj:
        obj_save_path = f"{cfg.results_dir}/{cfg.test_dataset}/{cfg.target.model}/{cfg.attack}/batch_{batch_id:02d}/objs"
        os.makedirs(obj_save_path, exist_ok=True)
        masker.save_obj_files(obj_save_path)
        logging.info(f"Successfully saved obj files to {obj_save_path}")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)

    # Print configuration for debugging
    logging.info("---------Configuration---------")
    logging.info(OmegaConf.to_yaml(cfg))
    logging.info("-------------------------------")

    # Create output directories
    results_dir = (
        f"{cfg.results_dir}/{cfg.test_dataset}/{cfg.target.model}/{cfg.attack}"
    )
    os.makedirs(results_dir, exist_ok=True)
    logging.info(f"Output directory created at: {results_dir}")

    # Configure dataloader
    root_dir = cfg.dataset[cfg.test_dataset].root_dir
    pair_list = cfg.dataset[cfg.test_dataset].pair_list[cfg.attack]
    logging.info(
        f"Loading {cfg.test_dataset} dataset from: {root_dir}, with pair list: {pair_list}"
    )
    if cfg.test_dataset == "celeba_hq":
        dataset = CelebAHQ(root_dir, pair_list)
        logging.info(f"CelebA-HQ dataset contains {len(dataset)} pairs.")
    elif cfg.test_dataset == "lfw":
        dataset = LFWPairs(root_dir, pair_list)
        logging.info(f"LFW dataset contains {len(dataset)} pairs.")
    else:
        raise ValueError(f"Dataset {cfg.test_dataset} not supported")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    logging.info(f"Dataloader created with batch size: {cfg.batch_size}")

    # Configure criterion
    if cfg.attack == "dodging":
        id_criterion = dodging_loss
    elif cfg.attack == "impersonation":
        id_criterion = impersonation_loss
    else:
        raise ValueError(f"Criterion {cfg.criterion} not supported")

    # Load surrogate models
    logging.info(f"Loading surrogate models:")
    surrogates = []
    for i in range(len(cfg.target.surrogates)):
        surrogates.append(FACE_MODELS[cfg.target.surrogates[i]])
        surrogates[i]["net"] = surrogates[i]["loader"](cfg.device).eval()

    # Configure wandb logger
    current_time = datetime.now().strftime("%Y%d%m_%H:%M:%S")
    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=f"{cfg.target.model}_{cfg.attack}_{cfg.test_dataset}_{current_time}",
        settings=wandb.Settings(x_disable_stats=True),
    )
    wandb.define_metric("run_step")
    wandb.define_metric("loss/*", step_metric="run_step")
    wandb.define_metric("similarity/*", step_metric="run_step")
    wandb.define_metric("images/*", step_metric="run_step")

    # Configure the Face Masker
    prn = PRN(*create_prnet(cfg))
    renderer = create_renderer(cfg.device)
    mask_textures = load_mask_textures(cfg.prn.mask_tex_path, cfg.prn.uv_face_mask)
    masker = FaceMasker(
        cfg.device,
        prn,
        renderer,
        list(mask_textures),
        is_aug=cfg.aug,
    )

    # Configure the Identity-Aware Style Injector
    style_injector = IdentityAwareStyleInjector(
        device=cfg.device,
        style_weight=cfg.style_weight,
        n_iter=cfg.style_n_iter,
        esb_size=cfg.target.esb_size[cfg.test_dataset],
        style_image_path=cfg.dataset.styles.root_dir,
        style_lr=cfg.style_lr,
    )
    style_injector.prepare_style_model(model_name=cfg.style_model)

    # Generate adversarial masks on surrogate models and save them
    n_batches = len(dataloader)
    for batch_id, batch_data in enumerate(dataloader):
        run_one_batch(
            masker,
            style_injector,
            batch_data,
            batch_id,
            n_batches,
            surrogates,
            id_criterion,
            cfg,
        )

        if cfg.debug:
            break

    style_injector.remove_hooks()

    logging.info("------------ Done! ------------")


if __name__ == "__main__":
    main()
