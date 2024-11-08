import text_recognizer.metadata.shared as shared

RAW_DATA_DIRNAME = shared.DATA_DIRNAME / "raw" / "iam"
METADATA_FILENAME = RAW_DATA_DIRNAME / "metadata.toml"
DL_DATA_DIRNAME = shared.DATA_DIRNAME / "downloaded" / "iam"
EXTRACTED_DATASET_DIRNAME = DL_DATA_DIRNAME / "iamdb"

DOWNSAMPLE_FACTOR = 2
LINE_REGION_PADDING = 8