from collections.abc import Callable
from text_recognizer.data.base_data_module import BaseDataModule, load_and_print_info
from text_recognizer.data.iam import IAM
from text_recognizer.data.utils import BaseDataset, convert_strings_to_label, resize_image
from text_recognizer.stems.paragraph import ParagraphStem
from typing import Dict, Tuple, Optional, Sequence

from PIL import Image

import argparse
import json
import numpy as np
import text_recognizer.metadata.iam_paragraphs as metadata

IMAGE_SCALE_FACTOR = metadata.IMAGE_SCALE_FACTOR
PROCESSED_DATA_DIRNAME = metadata.PROCESSED_DATA_DIRNAME
NEW_LINE_TOKEN = metadata.NEW_LINE_TOKEN

class IAMParagraph(BaseDataModule):
    """IAM handwriting database paragraphs."""

    def __init__(self, args: argparse.Namespace = None):
        super().__init__(args)
        
        self.augment = self.args.get("augment_data", "true").lower() == "true"
        
        self.mapping = metadata.MAPPING
        self.inverse_mapping = {v: k for k, v in enumerate(self.mapping)}
        
        self.input_dims = metadata.DIMS
        self.output_dims = metadata.OUTPUT_DIMS
        
        self.transform = ParagraphStem()
        self.trainval_transform = ParagraphStem(augment = self.augment)
        
    @staticmethod
    def add_to_argparse(parser: argparse.ArgumentParser) -> None:
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--augment_data", type = str, default = "true")
        return parser
    
    def prepare_data(self, *args, **kwargs) -> None:
        if (PROCESSED_DATA_DIRNAME / "_properties.json").exists():
            return 
        
        print("IAMParagraphs.prepare_data: Cropping IAM paragraph regions and saving them along with labels...")
        
        iam = IAM()
        iam.prepare_data()
        
        properties = {}
        
        for split in ["train", "val", "test"]:
            crops, labels = get_paragraph_crops_and_labels(iam, split)
            save_crops_and_labels(crops, labels, split)
            
            properties.update(
                {
                    id_: {
                        "crop_shape": crops[id_].size[::-1],
                        "label_length": len(label),
                        "num_lines": _num_lines(label)
                        
                    }
                    for id_, label in labels.items()
                }
            )
            
        with open(PROCESSED_DATA_DIRNAME / "_properties.json", "w") as file:
            json.dump(properties, file, indent = 4)
            
    def setup(self, stage: str = None) -> None:
        def _load_dataset(split: str, transform: Callable) -> BaseDataset:
            crops, labels = load_processed_crops_and_labels(split)
            Y = convert_strings_to_label(labels, self.inverse_mapping, length = self.output_dims[0])
            return BaseDataset(crops, Y, transform = transform)
        
        print(f"IAMParagraphs.setup({stage}): Loading IAM paragraph regions and lines...")
        
        validate_input_and_output_dimensions(input_dims = self.input_dims, output_dims = self.output_dims)
        
        if stage == "fit" or stage is None:
            self.train_dataset = _load_dataset(split = "train", transform = self.trainval_transform)
            self.val_dataset = _load_dataset(split = "val", transform = self.transform)
        
        if stage == "test" or stage is None:
            self.test_dataset = _load_dataset(split = "test", transform = self.transform)
    
    def __repr__(self) -> str:
        """Print info about the dataset."""
        basic = (
            "IAM Paragraphs Dataset\n"
            f"  Num classes: {len(self.mapping)}\n"
            f"  Input dims: {len(self.input_dims)}\n"
            f"  Output dims: {len(self.output_dims)}\n"
        )
        
        if self.train_dataset is None and self.val_dataset is None and self.train_dataset is None:
            return basic
        
        x, y = next(iter(self.train_dataloader()))
        xt, yt = next(iter(self.test_dataloader()))
        
        data = (
            f"  Train/val/test sizes: {len(self.train_dataset)}, {len(self.val_dataset)}, {len(self.test_dataset)}\n"
            f"  Train batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"  Train batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
            f"  Test batch x stats: {(xt.shape, xt.dtype, xt.min(), xt.mean(), xt.std(), xt.max())}\n"
            f"  Test batch y stats: {(yt.shape, yt.dtype, yt.min(), yt.max())}\n"
        )
        
        return basic + data
    
def validate_input_and_output_dimensions(
    input_dims: Optional[Tuple[int, ...]],
    output_dims: Optional[Tuple[int, ...]]
) -> None:
    """Validate input and output dimensions against the properties of the dataset."""
    properties = get_dataset_properties()
    
    max_image_shape = properties["crop_shape"]["max"] / IMAGE_SCALE_FACTOR
    
    assert input_dims is not None and input_dims[1] >= max_image_shape[0] and input_dims[2] >= max_image_shape[1]
    
    assert output_dims is not None and output_dims[0] >= properties["label_length"]["max"] + 2
    
def get_paragraph_crops_and_labels(
    iam: IAM, 
    split: str, 
    scale_factor: int = IMAGE_SCALE_FACTOR
) -> Tuple[Dict[str, Image.Image], Dict[str, str]]:
    """
    Create IAM paragraphs crops with resizing and their labels.

    Args:
        iam (IAM): IAM dataset.
        split (str): Dataset split used. Can be "train", "val", or "test".
        scale_factor (int, optional): factor by which the image is scaled. Defaults to IMAGE_SCALE_FACTOR.
    """
    
    crops = {}
    labels = {}
    
    for iam_id in iam.ids_from_split[split]:
        image = iam.load_image(iam_id)
        paragraph_region = iam.paragraph_region_from_id[iam_id]
        crops[iam_id] = image.crop([paragraph_region[i] for i in ["x1", "y1", "x2", "y2"]])
        crops[iam_id] = resize_image(crops[iam_id], scale_factor = scale_factor)
        labels[iam_id] = iam.paragraph_string_from_id[iam_id]
        
    assert len(crops) == len(labels)
    
    return crops, labels

def save_crops_and_labels(
    crops: Dict[str, Image.Image],
    labels: Dict[str, str],
    split: str
) -> None:
    """Save crops and their labels of a split."""
    (PROCESSED_DATA_DIRNAME / split).mkdir(parents = True, exist_ok = True)
    
    with open(_labels_filename(split), "w") as file:
        json.dump(labels, file, indent = 4)
    
    for id_, crop in crops.items():
        crop.save(_crop_filename(id_, split))
        
def load_processed_crops_and_labels(
    split: str
) -> Tuple[Sequence[Image.Image], Sequence[str]]:
    """Load processed crops and their labels of a split."""
    
    with open(_labels_filename(split), "r") as file:
        labels = json.load(file)
        
    sorted_ids = sorted(labels.keys())
    sorted_crops = [Image.open(_crop_filename(id_, split)).convert(mode = "L") for id_ in sorted_ids]
    sorted_labels = [labels[id_] for id_ in sorted_ids]
    
    assert len(sorted_crops) == len(sorted_labels)
    
    return sorted_crops, sorted_labels
    
def get_dataset_properties() -> dict:
    """Return properties describing the overall dataset."""
    
    with open(PROCESSED_DATA_DIRNAME / "_properties.json", "r") as file:
        properties = json.load(file)
        
    def _get_property_values(key: str) -> list:
        return [_[key] for _ in properties.values()]
    
    crop_shapes = np.array(_get_property_values("crop_shape"))
    aspect_ratios = crop_shapes[:, 1] / crop_shapes[:, 0]
    
    return {
        "label_length": {
            "min": min(_get_property_values("label_length")),
            "max": max(_get_property_values("label_length"))
        },
        "num_lines": {
            "min": min(_get_property_values("num_lines")),
            "max": max(_get_property_values("num_lines"))
        },
        "crop_shape": {
            "min": crop_shapes.min(axis = 0),
            "max": crop_shapes.max(axis = 0)
        },
        "aspect_ratio": {
            "min": aspect_ratios.min(), 
            "max": aspect_ratios.max()
        }
    }

def _labels_filename(split: str):
    """Return filename of the processed labels."""
    return PROCESSED_DATA_DIRNAME / split / "_labels.json"

def _crop_filename(id_: str, split: str):
    """Return filename of the processed crops."""
    return PROCESSED_DATA_DIRNAME / split / f"{id_}.png"

def _num_lines(label: str) -> int:
    """Return number of lines of text in label."""
    return label.count(NEW_LINE_TOKEN) + 1

if __name__ == "__main__":
    load_and_print_info(IAMParagraph)