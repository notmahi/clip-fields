import logging
import os
import pprint
import random
from typing import Dict, Union

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

import wandb
from dataloaders import (
    R3DSemanticDataset,
    DeticDenseLabelledDataset,
    ClassificationExtractor,
)
from misc import ImplicitDataparallel
from grid_hash_model import GridCLIPModel


SAVE_DIRECTORY = "clip_implicit_model"
DEVICE = "cuda"
IMAGE_TO_LABEL_CLIP_LOSS_SCALE = 1.0
LABEL_TO_IMAGE_LOSS_SCALE = 1.0
EXP_DECAY_COEFF = 0.5
SAVE_EVERY = 5
METRICS = {
    "accuracy": torchmetrics.Accuracy,
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def train(
    clip_train_loader: DataLoader,
    labelling_model: Union[GridCLIPModel, ImplicitDataparallel],
    optim: torch.optim.Optimizer,
    epoch: int,
    classifier: ClassificationExtractor,
    device: Union[str, torch.device] = DEVICE,
    exp_decay_coeff: float = EXP_DECAY_COEFF,
    image_to_label_loss_ratio: float = IMAGE_TO_LABEL_CLIP_LOSS_SCALE,
    label_to_image_loss_ratio: float = LABEL_TO_IMAGE_LOSS_SCALE,
    disable_tqdm: bool = False,
    metric_calculators: Dict[str, Dict[str, torchmetrics.Metric]] = {},
):
    total_loss = 0
    label_loss = 0
    image_loss = 0
    classification_loss = 0
    total_samples = 0
    total_classification_loss = 0
    labelling_model.train()
    total = len(clip_train_loader)
    for clip_data_dict in tqdm.tqdm(
        clip_train_loader,
        total=total,
        disable=disable_tqdm,
        desc=f"Training epoch {epoch}",
    ):
        xyzs = clip_data_dict["xyz"].to(device)
        clip_labels = clip_data_dict["clip_vector"].to(device)
        clip_image_labels = clip_data_dict["clip_image_vector"].to(device)
        image_weights = torch.exp(-exp_decay_coeff * clip_data_dict["distance"]).to(
            device
        )
        label_weights = clip_data_dict["semantic_weight"].to(device)
        image_label_index: torch.Tensor = (
            clip_data_dict["img_idx"].to(device).reshape(-1, 1)
        )
        language_label_index: torch.Tensor = (
            clip_data_dict["label"].to(device).reshape(-1, 1)
        )

        (predicted_label_latents, predicted_image_latents) = labelling_model(xyzs)
        # Calculate the loss from the image to label side.
        batch_size = len(image_label_index)
        image_label_mask: torch.Tensor = (
            image_label_index != image_label_index.t()
        ).float() + torch.eye(batch_size, device=device)
        language_label_mask: torch.Tensor = (
            language_label_index != language_label_index.t()
        ).float() + torch.eye(batch_size, device=device)

        # For logging purposes, keep track of negative samples per point.
        image_label_mask.requires_grad = False
        language_label_mask.requires_grad = False
        contrastive_loss_labels = labelling_model.compute_loss(
            predicted_label_latents,
            clip_labels,
            label_mask=language_label_mask,
            weights=label_weights,
        )
        contrastive_loss_images = labelling_model.compute_loss(
            predicted_image_latents,
            clip_image_labels,
            label_mask=image_label_mask,
            weights=image_weights,
        )
        del (
            image_label_mask,
            image_label_index,
            language_label_mask,
        )

        # Now figure out semantic segmentation.
        with torch.no_grad():
            class_probs = classifier.calculate_classifications(
                model_text_features=predicted_label_latents,
                model_image_features=predicted_image_latents,
            )
            # Now figure out semantic accuracy and loss.
            semseg_mask = torch.logical_and(
                language_label_index != -1,
                language_label_index < classifier.total_label_classes,
            ).squeeze(-1)
            if not torch.any(semseg_mask):
                classification_loss = torch.zeros_like(contrastive_loss_images)
            else:
                # Figure out the right classes.
                masked_class_prob = class_probs[semseg_mask]
                masked_labels = language_label_index[semseg_mask].squeeze(-1).long()
                classification_loss = F.cross_entropy(
                    torch.log(masked_class_prob),
                    masked_labels,
                )
                if metric_calculators.get("semantic"):
                    for _, calculators in metric_calculators["semantic"].items():
                        _ = calculators(masked_class_prob, masked_labels)

        contrastive_loss = (
            image_to_label_loss_ratio * contrastive_loss_images
            + label_to_image_loss_ratio * contrastive_loss_labels
        )

        optim.zero_grad(set_to_none=True)
        contrastive_loss.backward()
        optim.step()
        # Clip the temperature term for stability
        labelling_model.temperature.data = torch.clamp(
            labelling_model.temperature.data, max=np.log(100.0)
        )
        label_loss += contrastive_loss_labels.detach().cpu().item()
        image_loss += contrastive_loss_images.detach().cpu().item()
        total_classification_loss += classification_loss.detach().cpu().item()
        total_loss += contrastive_loss.detach().cpu().item()
        total_samples += 1

    to_log = {
        "train_avg/contrastive_loss_labels": label_loss / total_samples,
        "train_avg/contrastive_loss_images": image_loss / total_samples,
        "train_avg/semseg_loss": total_classification_loss / total_samples,
        "train_avg/loss_sum": total_loss / total_samples,
        "train_avg/labelling_temp": torch.exp(labelling_model.temperature.data.detach())
        .cpu()
        .item(),
    }
    for metric_dict in metric_calculators.values():
        for metric_name, metric in metric_dict.items():
            try:
                to_log[f"train_avg/{metric_name}"] = (
                    metric.compute().detach().cpu().item()
                )
            except RuntimeError as e:
                to_log[f"train_avg/{metric_name}"] = 0.0
            metric.reset()
    wandb.log(to_log)
    logger.debug(pprint.pformat(to_log, indent=4, width=1))
    return total_loss


def save(
    labelling_model: Union[ImplicitDataparallel, GridCLIPModel],
    optim: torch.optim.Optimizer,
    epoch: int,
    save_directory: str = SAVE_DIRECTORY,
    saving_dataparallel: bool = False,
):
    if saving_dataparallel:
        to_save = labelling_model.module
    else:
        to_save = labelling_model
    state_dict = {
        "model": to_save.state_dict(),
        "optim": optim.state_dict(),
        "epoch": epoch,
    }
    torch.save(
        state_dict,
        f"{save_directory}/implicit_scene_label_model_latest.pt",
    )
    return 0


def get_real_dataset(cfg):
    if cfg.use_cache:
        location_train_dataset = torch.load(cfg.saved_dataset_path)
    else:
        view_dataset = R3DSemanticDataset(cfg.dataset_path, cfg.custom_labels)
        if cfg.sample_freq != 1:
            view_dataset = Subset(
                view_dataset,
                torch.arange(0, len(view_dataset), cfg.sample_freq),
            )
        location_train_dataset = DeticDenseLabelledDataset(
            view_dataset,
            clip_model_name=cfg.web_models.clip,
            sentence_encoding_model_name=cfg.web_models.sentence,
            device=cfg.device,
            detic_threshold=cfg.detic_threshold,
            subsample_prob=cfg.subsample_prob,
            use_lseg=cfg.use_lseg,
            use_extra_classes=cfg.use_extra_classes,
            use_gt_classes=cfg.use_gt_classes_in_detic,
            visualize_results=cfg.visualize_detic_results,
            visualization_path=cfg.detic_visualization_path,
        )
        if cfg.cache_result:
            torch.save(location_train_dataset, cfg.cache_path)
    return location_train_dataset


@hydra.main(version_base="1.2", config_path="configs", config_name="train.yaml")
def main(cfg):
    seed_everything(cfg.seed)
    # Set up single thread tokenizer.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    real_dataset: DeticDenseLabelledDataset = get_real_dataset(cfg)
    # Setup our model with min and max coordinates.
    max_coords, _ = real_dataset._label_xyz.max(dim=0)
    min_coords, _ = real_dataset._label_xyz.min(dim=0)
    logger.debug(f"Environment bounds: max {max_coords} min {min_coords}")
    train_classifier = ClassificationExtractor(
        clip_model_name=cfg.web_models.clip,
        sentence_model_name=cfg.web_models.sentence,
        class_names=real_dataset._all_classes,
        device=cfg.device,
    )

    # Set up our metrics on this dataset.
    train_metric_calculators = {}
    train_class_count = {"semantic": train_classifier.total_label_classes}
    average_style = ["micro", "macro", "weighted"]
    for classes, counts in train_class_count.items():
        train_metric_calculators[classes] = {}
        for metric_name, metric_cls in METRICS.items():
            for avg in average_style:
                if "accuracy" in metric_name:
                    new_metric = metric_cls(
                        num_classes=counts, average=avg, multiclass=True
                    ).to(cfg.device)
                    train_metric_calculators[classes][
                        f"{classes}_{metric_name}_{avg}"
                    ] = new_metric

    if torch.cuda.device_count() > 1 and cfg.dataparallel:
        batch_multiplier = torch.cuda.device_count()
    else:
        batch_multiplier = 1

    clip_train_loader = DataLoader(
        real_dataset,
        batch_size=batch_multiplier * cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=cfg.num_workers,
    )
    logger.debug(f"Total train dataset sizes: {len(real_dataset)}")

    labelling_model = GridCLIPModel(
        image_rep_size=real_dataset[0]["clip_image_vector"].shape[-1],
        text_rep_size=real_dataset[0]["clip_vector"].shape[-1],
        mlp_depth=cfg.mlp_depth,
        mlp_width=cfg.mlp_width,
        log2_hashmap_size=cfg.log2_hashmap_size,
        num_levels=cfg.num_grid_levels,
        level_dim=cfg.level_dim,
        per_level_scale=cfg.per_level_scale,
        max_coords=max_coords,
        min_coords=min_coords,
    ).to(cfg.device)
    optim = torch.optim.Adam(
        labelling_model.parameters(),
        lr=cfg.lr,
        betas=tuple(cfg.betas),
        weight_decay=cfg.weight_decay,
    )

    save_directory = cfg.save_directory
    state_dict = "{}/implicit_scene_label_model_latest.pt".format(save_directory)

    if os.path.exists("{}/".format(save_directory)) and os.path.exists(state_dict):
        logger.info(f"Resuming job from: {state_dict}")
        loaded_dict = torch.load(state_dict)
        labelling_model.load_state_dict(loaded_dict["model"])
        optim.load_state_dict(loaded_dict["optim"])
        epoch = loaded_dict["epoch"]
        resume = "allow"
        del loaded_dict
    else:
        logger.info("Could not find old runs, starting fresh...")
        os.makedirs("{}/".format(save_directory), exist_ok=True)
        resume = False
        epoch = 0

    dataparallel = False
    if torch.cuda.device_count() > 1 and cfg.dataparallel:
        labelling_model = ImplicitDataparallel(labelling_model)
        dataparallel = True

    wandb.init(
        project=cfg.project,
        tags=[f"model/{cfg.model_type}"],
        config=OmegaConf.to_container(cfg, resolve=True),
        resume=resume,
    )
    # Set the extra parameters.
    wandb.config.web_labelled_points = len(real_dataset)

    # Disable tqdm if we are running inside slurm
    disable_tqdm = os.environ.get("SLURM_JOB_ID") is not None
    while epoch <= cfg.epochs:
        train(
            clip_train_loader,
            labelling_model,
            optim,
            epoch,
            train_classifier,
            cfg.device,
            exp_decay_coeff=cfg.exp_decay_coeff,
            image_to_label_loss_ratio=cfg.image_to_label_loss_ratio,
            label_to_image_loss_ratio=cfg.label_to_image_loss_ratio,
            disable_tqdm=disable_tqdm,
            metric_calculators=train_metric_calculators,
        )
        epoch += 1
        if epoch % SAVE_EVERY == 0:
            save(
                labelling_model,
                optim,
                epoch,
                save_directory=save_directory,
                saving_dataparallel=dataparallel,
            )


if __name__ == "__main__":
    main()
