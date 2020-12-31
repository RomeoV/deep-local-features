from absl import logging
from lib.tf_weight_loader import load_weights
from lib.tf_weight_loader import mapping as default_mapping
from torchvision import models

from absl import flags
from absl import app

flags.DEFINE_bool("test_print_tensors", True,
                  "Wether or not to print tensors for testing")
flags.DEFINE_bool("test_map_tensors", True,
                  "Wether or not to test the mapping")
flags.DEFINE_bool("test_mapping", True,
                  "Wether or not to run checks on default mapping")
flags.DEFINE_bool("test_model", True,
                  "Wether or not to test the model")

FLAGS = flags.FLAGS

dummy_mapping = {
    "test": "resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights"
}


def main(argv):
    weight_loader = load_weights.WeightLoader(mapping=dummy_mapping)

    if FLAGS.test_print_tensors:
        logging.info("Testing print tensor...")
        weight_loader.print_tensors()
        logging.info("Print tensor done!")
    if FLAGS.test_map_tensors:
        logging.info("Testing load variable...")
        weight_loader.load_variable("test")
        logging.info("Testing variable done!")
    if FLAGS.test_mapping:
        logging.info("Testing mapping...")
        model = models.resnet50()
        mapping = default_mapping.get_default_mapping()
        weight_loader = load_weights.WeightLoader(mapping=mapping)
        weight_loader.validate_mapping(model)
        logging.info("Testing mapping done!")
    if FLAGS.test_model:
        logging.info("Testing model...")
        model = models.resnet50()
        mapping = default_mapping.get_default_mapping()
        weight_loader = load_weights.WeightLoader(mapping=mapping)

        model = weight_loader.set_torch_model(model)
        logging.info("Testing model done!")


if __name__ == "__main__":
    app.run(main)
