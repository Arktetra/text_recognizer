from boltons.cacheutils import cachedproperty
from defusedxml import ElementTree
from pathlib import Path
from text_recognizer.data.base_data_module import _download_raw_dataset
from text_recognizer.metadata.iam_paragraphs import NEW_LINE_TOKEN
from text_recognizer.utils import temporary_working_directory, read_image_pil
from typing import cast, Any, Dict, List, Optional

from PIL import ImageOps

import text_recognizer.metadata.iam as metadata
import toml
import zipfile

METADATA_FILENAME = metadata.METADATA_FILENAME
DL_DATA_DIRNAME = metadata.DL_DATA_DIRNAME
EXTRACTED_DATASET_DIRNAME = metadata.EXTRACTED_DATASET_DIRNAME

class IAM:
    """
    A dataset of images of handwritten text written in a form underneath a typewritten prompt.
    """
    def __init__(self):
        self.metadata = toml.load(METADATA_FILENAME)
        
    def prepare_data(self):
        if self.xml_filenames:
            return
        filename = _download_raw_dataset(self.metadata, DL_DATA_DIRNAME)
        _extract_raw_dataset(filename, DL_DATA_DIRNAME)
        
    def load_images(self, id: str):
        """
        Load and return an image of an entire IAM form.
        
        The image is grayscale with white text on black background.
        
        The image contains printed prompt text at the top, above the handwritten text.
        
        The handwritten words, lines or paragraphs can be cropped out using the relevant crop region data.
        """
        image = read_image_pil(self.form_filenames_from_id[id], grayscale = True)
        image = ImageOps.invert(image)
        return image
        
    def __repr__(self):
        """Print info about the dataset."""
        info = ["IAM Dataset"]
        info.append(f"Total Images: {len(self.xml_filenames)}")
        info.append(f"Total Test Images: {len(self.test_ids)}")
        info.append(f"Total paragraphs: {len(self.paragraph_string_from_id)}")
        num_lines = sum(len(line_regions) for line_regions in self.line_regions_from_id.items())
        info.append(f"Total Lines: {num_lines}")
        
        return "\n\t".join(info)
                
    @cachedproperty
    def all_ids(self):
        """A list of all form IDs."""
        return [filename.stem for filename in self.xml_filenames]
    
    @cachedproperty
    def train_ids(self):
        """A list of form IDs from IAM LWITRT training set."""
        return list(set(self.all_ids) - (set(self.test_ids) | set(self.val_ids)))
    
    @cachedproperty
    def val_ids(self):
        """A list of form IDs from IAM LWITRT validation sets 1 and 2."""
        validation_ids = _get_ids_from_lwitlrt_split_file(EXTRACTED_DATASET_DIRNAME / "task" / "validationset1.txt")
        validation_ids.extend(_get_ids_from_lwitlrt_split_file(EXTRACTED_DATASET_DIRNAME / "task" / "validationset2.txt"))
        return validation_ids
            
    @cachedproperty
    def test_ids(self):
        """A list of form IDs from IAM LWITRT test set."""
        return _get_ids_from_lwitlrt_split_file(EXTRACTED_DATASET_DIRNAME / "task" / "testset.txt")
    
    @cachedproperty
    def ids_from_split(self):
        """
        A dictionary mapping splits to form IDS according to IAM Lines LWITLRT.
        """
        return {"train": self.train_ids, "val": self.val_ids, "test": self.test_ids}
    
    @cachedproperty
    def split_from_id(self):
        """
        A dictionary mapping form IDs to their split according to IAM Lines LWITLRT.
        """
        split_from_id = {id_: "train" for id_ in self.train_ids}
        split_from_id.update({id_: "val" for id_ in self.val_ids})
        split_from_id.update({id_: "test" for id_ in self.test_ids})
        return split_from_id

    @property
    def xml_filenames(self):
        """A list of the filenames of all .xml files which contain meta-information."""
        return list((EXTRACTED_DATASET_DIRNAME / "xml").glob("*.xml"))
    
    @property
    def xml_filenames_from_id(self):
        """
        A dicitonary mapping form IDs to their XML meta-information files.
        """
        return {filename.stem: filename for filename in self.xml_filenames}
    
    @property
    def form_filenames(self):
        """A list of the filenames of all .jpg files, which contain IAM form images."""
        return list((EXTRACTED_DATASET_DIRNAME / "forms").glob("*.jpg"))
    
    @property
    def form_filenames_from_id(self):
        """
        A dictionary mapping form IDS to their JPG images.
        """
        return {filename.stem: filename for filename in self.form_filenames}
    
    @cachedproperty 
    def line_strings_from_id(self):
        """
        A dict mapping an IAM form id to its list of handwritten line texts.
        """
        return {filename.stem: _get_line_strings_from_xml_file(filename) for filename in self.xml_filenames}
    
    @cachedproperty
    def paragraph_string_from_id(self):
        """
        A dict mapping an IAM form id to its handwritten paragraph text.
        """
        return {id_: NEW_LINE_TOKEN.join(line_string) for id_, line_string in self.line_strings_from_id.items()}
       
    @cachedproperty
    def line_regions_from_id(self):
        """
        A dict mapping an IAM form id to its list of handwritten line image crop regions.
        """
        return {filename.stem: _get_line_regions_from_xml_file(filename) for filename in self.xml_filenames}
    
    @cachedproperty
    def paragraph_region_from_id(self):
        """
        A dict mapping an IAM form id to its handwritten paragraph image crop region.
        """
        return {
            id: {
                "x1": min(region["x1"] for region in line_regions),
                "x2": max(region["x2"] for region in line_regions),
                "y1": min(region["y1"] for region in line_regions),
                "y2": max(region["y2"] for region in line_regions)
            }
            for id_, line_regions in self.line_regions_from_id.items()
        }
    
def _extract_raw_dataset(filename: Path, dirname: Path) -> None:
    """Extract the zipfile with `filename` in the `dirname` directoy."""
    print(f"Extracting {filename}")
    with temporary_working_directory(dirname):
        with zipfile.ZipFile(filename, "r") as zip_file:
            zip_file.extractall()
            
def _get_ids_from_lwitlrt_split_file(filename: Path) -> List[str]:
    """
    Get the ids from Large Writer Independent Text Line Recognition Task (LWITLRT) data split file.
    """
    with open(filename) as f:
        line_ids_str = f.read()
        
    line_ids = line_ids_str.split("\n")
    page_ids = list({"-".join(line_id.split("-")[:2]) for line_id in line_ids if line_id})
    return page_ids

def _get_line_strings_from_xml_file(filename: Path) -> List[str]:
    """
    Get the text content of each line. Note that &quot; is replaced with ".
    """
    xml_line_elements = _get_line_elements_from_xml_file(filename)
    return [_get_text_from_xml_element(el) for el in xml_line_elements]

def _get_line_elements_from_xml_file(filename: Path) -> List[Any]:
    """
    Get all line XML elements from XML file.
    """
    xml_root_element = ElementTree.parse(filename).getroot()
    return xml_root_element.findall("handwritten-part/line")

def _get_text_from_xml_element(xml_element) -> str:
    """
    Extract text from any XML element.
    """
    return xml_element.attrib["text"].replace('&quot;', '"')

def _get_line_regions_from_xml_file(filename: Path) -> Dict[str, int]:
    """
    Get dict mapping line to a region for each line.
    """
    xml_line_elements = _get_line_elements_from_xml_file(filename)
    
    line_regions = [
        cast(Dict[str, int], _get_region_from_xml_element(xml_elem = el, xml_path = "word/cmp")) for el in xml_line_elements
    ]
    
    assert any(region is not None for region in line_regions), "Line regions cannot be None."
    
    line_gaps_y = [
        max(next_line_region["y1"] - prev_line_region["y2"], 0) for next_line_region, prev_line_region in zip(line_regions[1:], line_regions[:-1])
    ]
    
    post_line_gaps_y = line_gaps_y + [2 * metadata.LINE_REGION_PADDING]
    pre_line_gaps_y = [2 * metadata.LINE_REGION_PADDING] + line_gaps_y
    
    return [
        {
            "x1": region["x1"] - metadata.LINE_REGION_PADDING,
            "x2": region["x2"] + metadata.LINE_REGION_PADDING,
            "y1": region["y1"] - min(metadata.LINE_REGION_PADDING, pre_line_gaps_y[i] // 2),
            "y2": region["y2"] - min(metadata.LINE_REGION_PADDING, post_line_gaps_y[i] // 2)
        }
        for i, region in enumerate(line_regions)
    ]

def _get_region_from_xml_element(xml_elem: Any, xml_path: str) -> Optional[Dict[str, int]]:
    """
    Get region from input xml element. The region is downsampled because the stored images are also downsampled.

    Args:
        xml_elem (Any): a line or word element with x, y, width, and height attributes.
        xml_path (str): "word/cmp" if xml_elem is a line element, else "cmp"

    Returns:
        Optional[Dict[str, int]]: _description_
    """
    unit_elements = xml_elem.findall(xml_path)
    
    if not unit_elements:
        return None
    
    return {
        "x1": min(int(el.attrib["x"]) for el in unit_elements) // metadata.DOWNSAMPLE_FACTOR,
        "y1": min(int(el.attrib["y"]) for el in unit_elements) // metadata.DOWNSAMPLE_FACTOR,
        "x2": max(int(el.attrib["x"]) + int(el.attrib["width"]) for el in unit_elements) // metadata.DOWNSAMPLE_FACTOR,
        "y2": max(int(el.attrib["y"]) + int(el.attrib["height"]) for el in unit_elements) // metadata.DOWNSAMPLE_FACTOR
    }

if __name__ == "__main__":
    iam = IAM()
    iam.prepare_data()
    print(iam)
    