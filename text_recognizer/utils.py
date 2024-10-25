from pathlib import Path
from typing import Union, Tuple, Any
from tqdm import tqdm
from urllib import request

from PIL import Image

import contextlib
import hashlib
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

def hasattrs(obj: Any, attrs: Tuple[str]) -> bool:
    """Checks if an object obj has all attribute in attrs.

    Args:
        obj (Any): an object.
        attrs (Tuple[str]): a tuple of attributes.

    Returns:
        bool: True if the condition holds, otherwise False.
    """
       
    return all(hasattr(obj, attr) for attr in attrs)

def to_categorical(y, num_classes):
    """1-hot encode a tensor."""
    return np.eye(num_classes, dtype = "uint8")[y]

def read_image_pil(image_uri: Union[Path, str], grayscale = False) -> Image:
    """Read PIL Image from an uri.

    Args:
        image_uri (Union[Path, str]): an image uri.
        grayscale (bool, optional): condition to grayscale the image. Defaults to False.

    Returns:
        Image: a PIL Image.
    """
    
    def read_image_from_filename(image_filename: Union[Path, str], grayscale: bool) -> Image:
        with Image.open(image_filename) as image:
            if grayscale:
                image = image.convert(model = "L")
            else:
                image = image.convert(mode = image.mode)
            return image
        
    def read_image_from_url(image_url: str, grayscale: bool) -> Image:
        url_response = request.urlopen(str(image_url))
        # image_array = np.array(bytearray(url_response.read()), dtype = np.uint8)
        return Image.open(url_response)
    
    local_file = os.path.exists(image_uri)
    
    try:
        img = None
        if local_file:
            img = read_image_from_filename(image_uri, grayscale)
        else:
            img = read_image_from_url(image_uri, grayscale)
    except Exception as e:
        raise ValueError("Could not load image at {}: {}".format(image_uri, e))

    return img
        
def read_image(image_uri: Union[Path, str], grayscale = False) -> np.ndarray:
    return read_image_pil(image_uri, grayscale)

def show_image(img: Union[np.ndarray, torch.Tensor], ax = None, figsize = None, title = None, noframe = True) -> plt.Axes:
    """Display an image from array or tensor.

    Args:
        img (Union[np.ndarray, torch.Tensor]): image in array or tensor form.
        figsize (_type_, optional): size of the figure. Defaults to None.
        title (_type_, optional): title of the figure. Defaults to None.
        noframe (bool, optional): condition to frame the figure. Defaults to True.

    Returns:
        plt.Axes: an axes in which the image is plotted.
    """
    
    if hasattrs(img, ("cpu", "detach", "permute")):
        img = img.detach().cpu()
        if len(img.shape) == 3 and img.shape[0] < 5:
            img = img.permute(1, 2, 0)
    elif not isinstance(img, np.ndarray):
        img = np.array(img)
    
    if img.shape[-1] == 1:
        img = img[..., 0]
    
    if ax is  None:
        _, ax = plt.subplots(figsize = figsize)
    
    ax.imshow(img)
    
    if title is not None:
        ax.set_title(title)
        
    ax.set_xticks([])
    ax.set_yticks([])
    
    if noframe:
        ax.axis("off")
        
    return ax

def show_images(sample: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], mapping = None, nrows = 1, ncols = 1, figsize = None):
    """Display a grid of images from the sample.

    Args:
        sample (Tuple[torch.Tensor, torch.Tensor]): sample from which image is displayed.
        mapping (_type_, optional): a mapping to actual labels. Defaults to None.
        nrows (int, optional): number of rows in the grid. Defaults to 1.
        ncols (int, optional): number of columns in the grid. Defaults to 1.
        figsize (_type_, optional): size of the figure. Defaults to None.
    """    
    if len(sample) == 2:
        imgs, labels = sample
    else:
        imgs, labels = sample, None
    
    if len(imgs.shape) == 4 and imgs.shape[1] < 5:
        imgs = imgs.permute(0, 2, 3, 1)
        
    if nrows == 1 and ncols == 1:
        title = None
        if mapping is not None:
            title = mapping[labels[0]]
        elif labels is not None:
            title = labels[0]
        return show_image(imgs[0], title = title)
    
    fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize = figsize)
    
    if nrows == 1 or ncols == 1:
        for idx in range(nrows * ncols):
            title = None
            if mapping is not None:
                title = mapping[labels[idx]]
            elif labels is not None:
                title = labels[idx]
            axes[idx] = show_image(img = imgs[idx], ax = axes[idx], title = title)
    else:
        plt.subplots_adjust(hspace = 0.25, wspace = -0.5)
        for row in range(nrows):
            for col in range(ncols):
                if (row * ncols + col) > (len(imgs) - 1):
                    axes[row][col].axis("off")
                    continue
                title = None
                if mapping is not None:
                    title = mapping[labels[row * ncols + col]]
                elif labels is not None:
                    title = labels[row * ncols + col]
                axes[row][col] = show_image(img = imgs[row * ncols + col], ax = axes[row][col], title = title)

@contextlib.contextmanager
def temporary_working_directory(working_dir: Union[Path, str]):
    """Temporarily switches to a directory, then returns to the original directory on exit."""
    curr_dir = os.getcwd()
    os.chdir(working_dir)
    try:
        yield
    finally:
        os.chdir(curr_dir)

def compute_sha256(filename: Union[Path, str]):
    """Returns SHA256 checksum of a file."""
    with open(filename, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

class TqdmUpTo(tqdm):
    
    def update_to(self, blocks: int = 1, bsize: int = 1, tsize: int = None):
        if tsize is not None:
            self.total = tsize
        self.update(blocks * bsize - self.n)
    
def download_url(url: str, filename: Union[Path, str]):
    """Download a file from url to filename, with a progress bar."""

    with TqdmUpTo(unit = "B", unit_scale = True, unit_divisor = 1024, miniters = 1) as t:
        request.urlretrieve(url, filename, reporthook = t.update_to, data = None)