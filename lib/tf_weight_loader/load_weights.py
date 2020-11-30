from typing import Optional
import tensorflow as tf
import numpy as np
from tensorflow.python.training import py_checkpoint_reader
import torch

from absl import flags
from absl import logging

flags.DEFINE_string("checkpoint_path", "/home/mark/Downloads/r50delg_gldv2clean_20200914/variables",
                    "Path to the 'varables' directory of the extracted weights. IMPORTANT: Add the checkpoint file before")

# Disclaimer: this code is largely adapted from the official [inspect_checkpoint.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/inspect_checkpoint.py)


class WeightLoader:
    def __init__(self, checkpoint: str, mapping={}) -> None:
        ckpt_path = tf.train.latest_checkpoint(checkpoint)
        self.reader = py_checkpoint_reader.NewCheckpointReader(ckpt_path)
        self.mapping = mapping

    def print_tensors(self) -> None:
        """Print all tensor variables in the checkpoint with their name and shape
        """
        var_to_shape_map = self.reader.get_variable_to_shape_map()
        var_to_dtype_map = self.reader.get_variable_to_dtype_map()
        count = 0
        for key, value in sorted(var_to_shape_map.items()):
            logging.info("tensor: %s (%s) %s" %
                         (key, var_to_dtype_map[key].name, value))
            count += 1
        logging.info(f"Found {count} variables")

    def load_variable(self, torch_name: str, dismiss_errors=True) -> Optional[np.ndarray]:
        """Tries to map the name of the torch variable to the tensorflow variable and then load it

        Args:
            torch_name (string): The name of the variable in the torch model
            dismiss_errors (bool, optional): Don't throw errors if the variable is not found in the mapping. Defaults to True.

        Returns:
            Optional[np.ndarray]: The weights as numpy array
        """
        if torch_name not in self.mapping:
            msg = f"Failed to find a mapping for {torch_name}"
            if dismiss_errors:
                logging.warn(msg)
                return None
            else:
                raise ValueError(msg)

        tf_name = self.mapping[torch_name]
        if not self.reader.has_tensor(tf_name):
            msg = f"Checkpoint has no entry for {tf_name}, mapped from {torch_name}"
            # I think it makes sense to always throw an error here. If we have the tensor in the mapping it should exist in the checkpoint file
            raise ValueError(msg)

        tensor = self.reader.get_tensor(tf_name)
        shape = tensor.shape
        dtype = tensor.dtype

        logging.info(f"Mapping: {torch_name} --> {tf_name} [{shape}, {dtype}]")

        # We need to convert it into torch shape
        if len(shape) == 4:
            return np.transpose(tensor, (3, 2, 0, 1))
        if len(shape) == 1:
            return tensor
        raise ValueError(
            "We did not expect a shape like this. If this was not an error, please implement it")

    def set_torch_model(self, torch_model, dismiss_errors=True):
        sd = torch_model.state_dict()
        for k in sd.keys():
            tf_weight = self.load_variable(k, dismiss_errors=dismiss_errors)
            if tf_weight is None:
                continue
            sd[k] = torch.from_numpy(tf_weight)
        torch_model.load_state_dict(sd)
        return torch_model

    def validate_mapping(self, torch_model) -> None:
        sd = torch_model.state_dict()
        for k, v in self.mapping.items():
            passed = True
            if k not in sd:
                passed = False
                logging.warn(
                    f"Failed to find pytorch variable {k} in state_dict")
            if not self.reader.has_tensor(v):
                passed = False
                logging.warn(f"Failed to find tf variable {v} in checkpoint")
            if passed:
                logging.info(f"Passed all tests for mapping {k} --> {v}")
