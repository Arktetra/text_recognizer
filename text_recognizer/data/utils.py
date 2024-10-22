from typing import Union, Sequence, Callable, Tuple, Any

from PIL import Image

import torch

SequenceOrTensor = Union[Sequence, torch.Tensor]

class BaseDataset(torch.utils.data.Dataset):
    """Base Dataset class that processes data and targets through optional transforms.

    Args:
        data (SequenceOrTensor): torch tensors, numpy arrays or PIL Images
        targets (SequenceOrTensor): torch tensors or numpy arrays.
        transform (Callable, optional): function that transforms a datum. Defaults to None.
        target_transform (Callable, optional): function that transforms a target. Defaults to None.
        
    """
    
    def __init__(
        self,
        data: SequenceOrTensor,
        targets: SequenceOrTensor,
        transform: Callable = None,
        target_transform: Callable = None
    ) -> None:
        if len(data) != len(targets):
            raise ValueError("Data and targets must be of equal length")
        
        super().__init__()
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform 
        
    def __len__(self) -> int:
        """Return length of the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Return a datum and its target after processing by transforms."""
        
        datum, target = self.data[idx], self.targets[idx]
        
        if self.transform is not None:
            datum = self.transform(datum)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return (datum, target)
    
def split_dataset(base_dataset: BaseDataset, fraction: float, seed: int) -> Tuple[BaseDataset, BaseDataset]:
    """
    Split input base_dataset into two base_datasets, the first with size 
    `fraction * size of the base_dataset` and the other with size `(1 - size of
    the base_dataset)`.
    """
    split_first_size = int(fraction * len(base_dataset))
    split_second_size = len(base_dataset) - split_first_size
    
    return torch.utils.data.random_split(
        base_dataset, [split_first_size, split_second_size], generator = torch.Generator().manual_seed(seed)
    )
    
def resize_image(image: Image.Image, scale_factor: int) -> Image.Image:
    """Resize image by scale factor."""
    if scale_factor == 1:
        return image
    
    return image.resize((image.width // scale_factor, image.height // scale_factor), resample = Image.BILINEAR)