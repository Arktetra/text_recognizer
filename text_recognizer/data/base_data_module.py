from .utils import BaseDataset
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Collection, Dict, Tuple, Optional

import argparse
import os
import text_recognizer.metadata.shared as metadata
import torch
import utils

def load_and_print_info(data_module_class) -> None:
    parser = argparse.ArgumentParser()
    data_module_class.add_to_argparse(parser)
    args = parser.parse_args()
    dataset = data_module_class(args)
    dataset.prepare_data()
    dataset.setup()
    print(dataset)
    
def _download_raw_dataset(metadata: Dict, dl_dirname: Path) -> Path:
    dl_dirname.mkdir(parents = True, exist_ok = True)
    filename = dl_dirname / metadata["filename"]
    
    if filename.exists():
        return filename
    
    print(f"Downloading raw dataset from {metadata['url']} to {filename}...")
    utils.download_url(metadata["url"], filename)
    print("Computing SHA-256...")
    sha256 = utils.compute_sha256(filename)
    if sha256 != metadata["sha256"]:
        raise ValueError("Downloaded data file SHA-256 does not match that listed in metadata document.")
    
    return filename

BATCH_SIZE = 128
NUM_AVAIL_CPUS = os.cpu_count()
NUM_AVAIL_GPUS = torch.cuda.device_count()

DEFAULT_NUM_WORKERS = NUM_AVAIL_CPUS

class BaseDataModule:
    def __init__(self, args: argparse.Namespace = None) -> None:
        self.args = vars(args) if args is not None else {}
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.num_workers = self.args.get("num_workers", DEFAULT_NUM_WORKERS)
        
        self.on_gpu = isinstance(self.args.get("gpus", None), (str, int))
        
        # Set the variables below in subclasses
        self.input_dims: Tuple[int, ...]
        self.output_dims: Tuple[int, ...]
        self.mapping: Collection
        self.train_dataset: BaseDataset
        self.val_dataset: BaseDataset
        self.test_dataset: BaseDataset
        
    @classmethod
    def data_dirname(cls):
        return metadata.DATA_DIRNAME
        
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--batch_size",
            type = int,
            default = BATCH_SIZE,
            help = f"Number of examples to operate on per forward step. Defaults to {BATCH_SIZE}"
        )    
        
        parser.add_argument(
            "--num_workers",
            type = int,
            default = DEFAULT_NUM_WORKERS,
            help = f"Number of additional processes to load data. Defaults to {DEFAULT_NUM_WORKERS}"
        )
        
        return parser
    
    def config(self):
        """
        Return import settings of the dataset, which will be passed to instantiate models.
        """
        return {
            "input_dims": self.input_dims,
            "output_dims": self.output_dims,
            "mapping": self.mapping
        }
        
    def prepare_data(self, *args, **kwargs) -> None:
        pass
    
    def setup(self, stage: Optional[str] = None) -> None:
        pass
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = self.num_workers,
            pin_memory = self.on_gpu
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = self.num_workers,
            pin_memory = self.on_gpu
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = self.num_workers,
            pin_memory = self.on_gpu
        )