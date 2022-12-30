import json
import os
import warnings
from contextlib import nullcontext
from datetime import datetime
from time import time
from typing import Tuple

import numpy as np
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DeformableDetrConfig, DeformableDetrForObjectDetection

from doclaynet import DocLayNet

warnings.filterwarnings("ignore", category=FutureWarning)


def finetune(
    train_dataset: DocLayNet,
    val_dataset: DocLayNet,
    model_dir: str = "SenseTime/deformable-detr",
    image_size: Tuple[int, int] = (800, 800),
    batch_size: int = 1,
    epochs: int = 10000,
    patience: int = 10,
    lr: float = 1e-4,
    lr_backbone: float = 1e-5,
    weight_decay: float = 1e-4,
    grad_clip_value: float = 0.1,
):
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        collate_fn=val_dataset.collate_fn,
    )

    dataloaders = {"train": train_dataloader, "val": val_dataloader}

    print(
        "Train Batch Cnt = {} / Test Batch Cnt {}".format(
            len(train_dataloader),
            len(val_dataloader),
        )
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    id2label = {str(k - 1): v["name"] for k, v in train_dataset.coco.cats.items()}

    if model_dir is None:
        with open("model/deformable-detr/config.json", "r") as fi:
            config_json = json.load(fi)

        config_json["id2label"] = {k: v for k, v in id2label.items()}
        config_json["label2id"] = {v: k for k, v in id2label.items()}

        config = DeformableDetrConfig(**config_json)
        model = DeformableDetrForObjectDetection(config).to(device)

    else:
        model = DeformableDetrForObjectDetection.from_pretrained(
            model_dir,
            num_labels=len(id2label),
            ignore_mismatched_sizes=True,
        ).to(device)

    optimizer_params = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": lr_backbone,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=lr,
        weight_decay=weight_decay,
    )

    time_stamp = str(datetime.now()).split(".")[0].replace(" ", "_")
    log_file = open(f"training_log-{time_stamp}.txt", "w")
    log_file.write(
        f"model_dir = {model_dir}\n"
        + f"image_size = {image_size}\n"
        + f"lr = {lr}\n"
        + f"lr_backbone = {lr_backbone}\n"
        + f"weight_decay = {weight_decay}\n"
        + f"grad_clip_value = {grad_clip_value}\n"
        + f"batch_size = {batch_size}\n"
        + f"patience = {patience}\n"
        + f"train batch cnt = {len(train_dataloader)}\n"
        + f"val batch cnt = {len(val_dataloader)}\n"
    )
    log_file.flush()

    best_loss = np.inf
    curr_patience = 0

    for epoch in range(epochs):
        for phase in ["train", "val"]:

            if phase == "train":
                model.train()
                context_manager = nullcontext()

            else:
                model.eval()
                context_manager = torch.no_grad()

            (
                total_loss,
                total_loss_ce,
                total_loss_bbox,
                total_loss_giou,
                total_card_err,
            ) = (0, 0, 0, 0, 0)

            start_time = time()

            pbar = tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase]))
            with context_manager:
                for i, batch in pbar:
                    pbar.set_description(
                        "{} {}: Loss {:.4f} - CE Loss {:.4f} - Bbox Loss {:.4f} - GIoU Loss {:.4f} - Card Err {:.4f} - Time {:.2f}sec".format(
                            epoch + 1,
                            phase,
                            total_loss / i if i > 0 else 0,
                            total_loss_ce / i if i > 0 else 0,
                            total_loss_bbox / i if i > 0 else 0,
                            total_loss_giou / i if i > 0 else 0,
                            total_card_err / i if i > 0 else 0,
                            time() - start_time,
                        )
                    )

                    pixel_values = batch["pixel_values"].to(device)
                    pixel_mask = batch["pixel_mask"].to(device)
                    labels = [
                        {k: v.to(device) for k, v in label.items()}
                        for label in batch["labels"]
                    ]

                    outputs = model(
                        pixel_values=pixel_values,
                        pixel_mask=pixel_mask,
                        labels=labels,
                    )

                    loss = outputs.loss

                    if phase == "train":
                        # clip_grad_norm_(model.parameters(), grad_clip_value)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                    total_loss += loss.item()
                    total_loss_ce += outputs.loss_dict["loss_ce"].item()
                    total_loss_bbox += outputs.loss_dict["loss_bbox"].item()
                    total_loss_giou += outputs.loss_dict["loss_giou"].item()
                    total_card_err += outputs.loss_dict["cardinality_error"].item()

                    if i == len(dataloaders[phase]) - 1:
                        log_file.write(
                            "{} {}: Loss {:.4f} - CE Loss {:.4f} - Bbox Loss {:.4f} - GIoU Loss {:.4f} - Card Err {:.4f} - Time {:.2f}sec\n".format(
                                epoch + 1,
                                phase,
                                total_loss / (i + 1),
                                total_loss_ce / i + 1,
                                total_loss_bbox / (i + 1),
                                total_loss_giou / (i + 1),
                                total_card_err / (i + 1),
                                time() - start_time,
                            )
                        )
                        log_file.flush()

                        if phase == "val":
                            epoch_loss = total_loss / len(dataloaders[phase])

                            if epoch_loss < best_loss:
                                os.makedirs("model/train", exist_ok=True)
                                torch.save(model.state_dict(), "model/train")

                                pbar.set_description(
                                    "{} {}: Best loss improved from {:.4f} to {:.4f}".format(
                                        epoch + 1,
                                        phase,
                                        best_loss,
                                        epoch_loss,
                                    ),
                                    refresh=True,
                                )

                                best_loss = epoch_loss
                                curr_patience = 0

                            else:
                                curr_patience += 1

                                pbar.set_description(
                                    "{} {}: Best loss not improved from {:.4f} to {:.4f}".format(
                                        epoch + 1,
                                        phase,
                                        best_loss,
                                        epoch_loss,
                                    ),
                                    refresh=True,
                                )

                            if curr_patience == patience:
                                log_file.close()
                                return

    log_file.close()
    return


if __name__ == "__main__":
    doclaynet_dir = "/mnt/d/DocLayNet"

    batch_size = 6

    image_height = 480
    image_width = int((image_height * 8.5) / 11)

    image_size = (image_height, image_width)

    train_dataset = DocLayNet(doclaynet_dir, part="train", image_size=image_size)
    val_dataset = DocLayNet(doclaynet_dir, part="val", image_size=image_size)

    finetune(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        image_size=image_size,
        batch_size=batch_size,
    )
