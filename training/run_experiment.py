
import argparse
import numpy as np
import torch

from utils import DATA_CLASS_MODULE, MODEL_CLASS_MODULE, import_class, setup_data_and_model_from_args

def _setup_parser():
    """Set up Python's ArgumentParser with data, model, and other arguments."""
    parser = argparse.ArgumentParser(add_help = False)
    
    parser.add_argument(
        "--data_class",
        type = str,
        default = "MNIST",
        help = f"String identifier for the data class, relative to {DATA_CLASS_MODULE}."
    )
    
    parser.add_argument(
        "--model_class",
        type = str,
        default = "CNN",
        help = f"String identifier for the model class, relative to {MODEL_CLASS_MODULE}"
    )
    
    parser.add_argument(
        "--load_checkpoint",
        type = str,
        default = None,
        help = "If passes, loads a model from the provided path."
    )
    
    parser.add_argument(
        "--stop_early",
        type = int,
        default = 0,
        help = "If non-zero, applies early stopping, with provided value as the patience argument."
        + "Defaults to 0."
    )
    
    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    data_class = import_class(f"{DATA_CLASS_MODULE}.{temp_args.data_class}")
    model_class = import_class(f"{MODEL_CLASS_MODULE}.{temp_args.model_class}")
    
    # Get the data and model
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)
    
    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)
    
    parser.add_argument("--help", "-h", action = "help")
    
    return parser 

def main():
    parser = _setup_parser()
    args = parser.parse_args()
    print(args)
    # data, model = setup_data_and_model_from_args(args)
    
if __name__ == "__main__":
    main()