from text_recognizer.data.base_data_module import BaseDataModule, load_and_print_info
from text_recognizer.stems.image import MNISTStem
from torch.utils.data import random_split
from torchvision.datasets import MNIST as TorchMNIST

import argparse
import text_recognizer.metadata.mnist as metadata

class MNIST(BaseDataModule):
    
    def __init__(self, args: argparse.Namespace) -> None:
        self.data_dir = metadata.DOWNLOADED_DATA_DIRNAME
        self.transform = MNISTStem()
        self.input_dims = metadata.DIMS
        self.output_dims = metadata.OUTPUT_DIMS
        self.mapping = metadata.MAPPING
        
    def prepare_data(self, *args, **kwargs) -> None:
        TorchMNIST(self.data_dir, train = True, download = True)
        TorchMNIST(self.data_dir, train = False, download = True)
        
    def setup(self, stage = None) -> None:
        mnist_full = TorchMNIST(self.data_dir, train = True, transform = self.transform)
        self.train_dataset, self.val_dataset = random_split(mnist_full, [metadata.TRAIN_SIZE, metadata.VAL_SIZE])
        self.test_dataset = TorchMNIST(self.data_dir, train = False, transform = self.transform)        
        
if __name__ == "__main__":
    load_and_print_info(MNIST)