from collections import defaultdict
from text_recognizer.data.base_data_module import BaseDataModule
from text_recognizer.data.emnist import EMNIST
from text_recognizer.data.sentence_generator import SentenceGenerator
from text_recognizer.data.utils import BaseDataset
from text_recognizer.stems.image import ImageStem
from typing import Dict, List, Tuple, Sequence

import argparse
import h5py
import numpy as np
import text_recognizer.metadata.emnist_lines as metadata
import torch

PROCESSED_DATA_DIRNAME = metadata.PROCESSED_DATA_DIRNAME
ESSENTIALS_FILENAME = metadata.ESSENTIALS_FILENAME

DEFAULT_MAX_LENGTH = 32
DEFAULT_MIN_OVERLAP = 0
DEFAULT_MAX_OVERLAP = 0.33

NUM_TRAIN = 10000
NUM_VAL = 2000
NUM_TEST = 2000

class EMNISTLines(BaseDataModule):
    """EMNIST Lines dataset: synthetic handwriting lines dataset made from EMNIST characters."""
    
    def __init__(self, args: argparse.Namespace = None):
        super().__init__(args)
        
        self.max_length = self.args.get("max_length", DEFAULT_MAX_LENGTH)
        self.min_overlap = self.args.get("min_overlap", DEFAULT_MIN_OVERLAP)
        self.max_overlap = self.args.get("max_overlap", DEFAULT_MAX_OVERLAP)
        self.num_train = self.args.get("num_train", NUM_TRAIN)
        self.num_val = self.args.get("num_val", NUM_VAL)
        self.num_test = self.args.get("num_test", NUM_TEST)
        
        self.mapping = metadata.MAPPING
        self.output_dims = (self.max_length, 1)
        max_width = self.max_length * metadata.CHAR_WIDTH
        self.input_dims = (*metadata.DIMSL[:2], max_width)
        
        self.emnist = EMNIST()

        self.transform = ImageStem()
        
    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        
        parser.add_argument(
            "--max_length",
            type = int,
            default = DEFAULT_MAX_LENGTH,
            help = f"Max line length in characters. Defaults to {DEFAULT_MAX_LENGTH}"
        )
        
        parser.add_argument(
            "--min_overlap",
            type = int,
            default = DEFAULT_MIN_OVERLAP,
            help = f"Min overlap between characters in a line, between 0 and 1. Defaults to {DEFAULT_MIN_OVERLAP}"
        )
        
        parser.add_argument(
            "--max_overlap",
            type = int,
            default = DEFAULT_MAX_OVERLAP,
            help = f"Max overlap between characters in a line, between 0 and 1. Defaults to {DEFAULT_MAX_OVERLAP}"
        )
        
        parser.add_argument(
            "--with_start_end_tokens",
            action = "store_true",
            default = False
        )
        
    @property
    def data_filename(self):
        return (
            PROCESSED_DATA_DIRNAME 
            / f"ml_{self.max_length}_o{self.min_overlap:f}_{self.max_overlap:f}_ntr{self.num_train}_nvl{self.num_val}_nte{self.num_test}_{self.with_start_end_tokens}"
        )
        
    def prepare_data(self, *args, **kwargs) -> None:
        if self.data_filename.exists():
            return 
        
        np.random.seed(42)
        self._generate_data("train")
        self._generate_data("val")
        self._generate_data("test")
        
    def setup(self, stage: str = None) -> None:
        print("EMNISTLines: loading data from HDF5...")
        
        if stage == "fit" or stage is None:
            with h5py.File(self.data_filename, "r") as f:
                x_train = f["x_train"][:]
                y_train = f["y_train"][:].astype(int)
                x_val = f["x_val"][:]
                y_val = f["y_val"][:].astype(int)
                
            self.train_dataset = BaseDataset(x_train, y_train, transform = self.transform)
            self.val_dataset = BaseDataset(x_val, y_val, self.transform)
            
        if stage == "test" or stage is None:
            with h5py.File(self.data_filename, "r") as f:
                x_test = f["x_test"][:]
                y_test = f["y_test"][:].astype(int)
            
            self.test_dataset = BaseDataset(x_test, y_test, transform = self.transform)
                
    
    def _generate_data(self, split: str) -> None:
        print(f"EMNISTLines: generating data for {split}...")

        sentence_generator = SentenceGenerator(self.max_length - 2) # two is subtracted for adding start and end tokens
        
        emnist = self.emnist
        emnist.prepare_data()
        emnist.setup()
        
        if split == "train":
            samples_of_char = get_samples_of_char(emnist.x_trainval, emnist.y_trainval)
            num = self.num_train
        elif split == "val":
            samples_of_char = get_samples_of_char(emnist.x_trainval, emnist.y_trainval, self.mapping)
            num = self.num_val
        else:
            samples_of_char = get_samples_of_char(emnist.x_test, emnist.y_test, emnist.mapping)
            num = self.num_test
            
        PROCESSED_DATA_DIRNAME.mkdir(parents = True, exist_ok = True)
        
        with h5py.File(self.data_filename, "a") as f:
            x, y = create_dataset_of_images(
                N = num,
                samples_of_char = samples_of_char,
                sentence_generator = sentence_generator,
                min_overlap = self.min_overlap,
                max_overlap = self.max_overlap,
                dims = self.input_dims
            )
            
            y = convert_strings_to_label(
                strings = y,
                mapping = emnist.inverse_mapping,
                length = self.output_dims[0],
                with_start_end_tokens = self.with_start_end_tokens
            )
            
            f.create_dataset(f"x_{split}", data = x, dtype = "u1", compression = "1zf")
            f.create_dataset(f"y_{split}", data = y, dtype = "u1", compression = "lzf")
            
    def __repr__(self) -> str:
        """Print info about the dataset."""
        
        basic = (
            "EMNIST Lines Dataset\n"
            f"Min overlap: {self.min_overlap}\n"
            f"Max overlap: {self.max_overlap}\n"
            f"Num classes: {len(self.mapping)}\n"
            f"Dims: {self.input_dims}\n"
            f"Output dims: {self.output_dims}\n"
        )
        
        if self.train_dataset is None and self.val_dataset is None and self.test_dataset is None:
            return basic 
        
        x, y = next(iter(self.train_dataloader()))
        
        data = (
            f"Train/val/test sizes: {len(self.train_dataset)}, {len(self.val_dataset)}, {len(self.test_dataset)}\n"
            f"Batch x stats: {(x.shape, x.dtype, x.min().item(), x.mean().item(), x.std().item(), x.max().item())}\n"
            f"Batch y stats: {(y.shape, y.dtype, y.min().item(), y.max().item())}"
        )
        
        return basic + data
        

def get_samples_of_char(samples: torch.Tensor, labels: torch.Tensor, mapping: List[str]) -> Dict[str, List[torch.Tensor]]:
    """Get all the sample images for a character from EMNIST dataset.

    Args:
        samples (torch.Tensor): image samples.
        labels (torch.Tensor): labels corresponding to the image samples.

    Returns:
        Dict[str, List[torch.Tensor]]: dictionary of characters and all the sample images corresponding to those characters.
    """
    samples_from_char = defaultdict(list)
    
    for sample, label in zip(samples, labels):
        samples_from_char(mapping[label]).append(sample)
        
    return samples_from_char

def select_letter_samples_for_string(string: str, samples_of_char: Dict[str, List[torch.Tensor]], char_shape: Tuple[int, int] = (metadata.CHAR_HEIGHT, metadata.CHAR_WIDTH)) -> List[torch.Tensor]:
    """Select sample from the available samples for each unique character in a string.

    Args:
        string (str): a string.
        samples_of_char (Dict[str, List[torch.Tensor]]): dictionary of characters and all the sample images corresponding to those characters.
        char_shape (Tuple[int, int], optional): shape of each character image. Defaults to (metadata.CHAR_HEIGHT, metadata.CHAR_WIDTH).

    Returns:
        List[torch.Tensor]: list of image samples forming the string.
    """
    zero_image = torch.zeros(char_shape, dtype = torch.uint8)
    sample_image_of_char = {}
    
    for char in string:
        if char in sample_image_of_char:
            continue
        samples = samples_of_char[char]
        sample = samples[np.random.choice(len(samples))] if samples else zero_image
        sample_image_of_char[char] = sample.reshape(*char_shape)
        
    return [sample_image_of_char[char] for char in string]

def construct_image_from_string(string: str, samples_of_char: Dict[str, List[torch.Tensor]], min_overlap: float, max_overlap: float, width: int) -> torch.Tensor:
    """Construct image of a string.

    Args:
        string (str): a string.
        samples_of_char (Dict[str, List[torch.Tensor]]): dictionary of characters and all the sample images corresponding to those characters.
        min_overlap (float): minimum value that overlap can take.
        max_overlap (float): maximum value that overlap can take.
        width (int): width of the image.

    Returns:
        torch.Tensor: image formed by concatenating all the character images.
    """
    overlap = np.random.uniform(min_overlap, max_overlap)
    sampled_images = select_letter_samples_for_string(string, samples_of_char)
    H, W = sampled_images[0].shape
    next_overlap_width = W - int(W * overlap)
    concatenated_image = torch.zeros((H, width), dtype = torch.uint8)
    x = 0
    
    
    for image in sampled_images:
        concatenated_image[:, x : (x + W)] += image
        x += next_overlap_width
    
    return torch.minimum(torch.Tensor([255]), concatenated_image)

def create_dataset_of_images(N: int, samples_of_char: Dict[str, List[torch.Tensor]], sentence_generator: SentenceGenerator, min_overlap: float, max_overlap: float, dims: int) -> Tuple[torch.Tensor, List[str]]:
    """Create a dataset of images for labels generated by the sentence generator.

    Args:
        N (int): number of items in the dataset.
        samples_of_char (Dict[str, List[torch.Tensor]]): dictionary of characters and all the sample images corresponding to those characters.
        sentence_generator (SentenceGenerator): an instance of SentenceGenerator.
        min_overlap (float): minimum value that overlap can take.
        max_overlap (float): maximum value that overlap can take.
        dims (int): dimension of each image in the dataset.

    Returns:
        Tuple[torch.Tensor, List[str]]: a tuple containing a tensor of images and a list containing corresponding labels.
        
    """
    imgs = torch.zeros(N, dims[1], dims[2])
    labels = []
    
    for n in range(N):
        label = sentence_generator.generate()
        imgs[n] = construct_image_from_string(label, samples_of_char, min_overlap, max_overlap, dims[-1])
        labels.append(label)
    
    return imgs, labels

def convert_strings_to_label(strings: Sequence[str], mapping: Dict[str, int], length: int, with_start_end_tokens: bool) -> np.ndarray:
    """Convert strings into labels of shape `(len(strings), length)` after adding 
    padding token <P> in the vacant spaces.

    Args:
        strings (Sequence[str]): a sequence of strings generated by the sentence generator.
        mapping (Dict[str, int]): an inverse one-to-one mapping of characters to integers.
        length (int): the length (number of characters) each image can have.
        with_start_end_tokens (bool): add start token <S> and end token <E> before conversion.

    Returns:
        np.ndarray: label for the strings.
    """
    labels = np.ones((len(strings), length)) * mapping["<P>"]
    
    for i, string in enumerate(strings):
        tokens = list(string)
        
        if with_start_end_tokens:
            tokens = ["<S>", *tokens, "<E>"]
        
        for ii, token in enumerate(tokens):
            labels[i][ii] = mapping[token]
            
    return labels