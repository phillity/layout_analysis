from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.datasets import CocoDetection
from transformers import DeformableDetrImageProcessor

"""
DocLayNet: A Large Human-Annotated Dataset for Document-Layout Analysis
https://arxiv.org/abs/2206.01062
https://github.com/DS4SD/DocLayNet

PubLayNet: Largest Dataset Ever for Document-Layout Analysis
https://arxiv.org/abs/1908.07836
https://github.com/ibm-aur-nlp/PubLayNet
"""


class DocLayNet(CocoDetection):
    def __init__(
        self,
        doclaynet_dir: str,
        part: str = "train",
        image_size: Tuple[int, int] = (800, 800),
    ):
        self.doclaynet_dir = doclaynet_dir
        self.part = part

        self.ann_file = f"{self.doclaynet_dir}/COCO/{part}.json"
        self.img_dir = f"{self.doclaynet_dir}/PNG"

        self.image_size = image_size

        super(DocLayNet, self).__init__(self.img_dir, self.ann_file)

        self.image_processor = DeformableDetrImageProcessor.from_pretrained(
            "SenseTime/deformable-detr"
        )

    def __getitem__(self, idx: int):
        image, target = super(DocLayNet, self).__getitem__(idx)

        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        encoding = self.image_processor(
            images=image,
            annotations=target,
            return_tensors="pt",
            size=self.image_size,
        )

        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]
        target["class_labels"] = target["class_labels"] - 1

        return pixel_values, target

    def collate_fn(
        self, batch: List[Tuple[torch.tensor, torch.tensor]]
    ) -> Dict[str, torch.tensor]:
        pixel_values = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        encoding = self.image_processor.pad_and_create_pixel_mask(
            pixel_values, return_tensors="pt"
        )

        batch = {
            "pixel_values": encoding["pixel_values"],
            "pixel_mask": encoding["pixel_mask"],
            "labels": labels,
        }

        return batch

    def show_sample_image(self) -> Image:
        image_ids = self.coco.getImgIds()
        image_id = image_ids[np.random.randint(0, len(image_ids))]

        image = self.coco.loadImgs(image_id)[0]
        image = Image.open(self.img_dir + "/" + image["file_name"])

        annotations = self.coco.imgToAnns[image_id]

        draw = ImageDraw.Draw(image, "RGBA")

        cats = self.coco.cats
        id2label = {k: v["name"] for k, v in cats.items()}

        for annotation in annotations:
            class_idx = annotation["category_id"]
            box = annotation["bbox"]
            x, y, w, h = tuple(box)

            draw.rectangle((x, y, x + w, y + h), outline="red", width=1)
            draw.text((x, y), id2label[class_idx], fill="white")

        image = image.resize(self.image_size[::-1])

        return image
