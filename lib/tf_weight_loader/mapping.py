
from typing import Mapping


def add_batchnorm(mapping: dict, torch_prefix: str, tf_prefix: str) -> dict:
    tf_prefix = f"{tf_prefix}/BatchNorm"

    mapping[f"{torch_prefix}.weight"] = f"{tf_prefix}/gamma"
    mapping[f"{torch_prefix}.bias"] = f"{tf_prefix}/beta"
    mapping[f"{torch_prefix}.running_mean"] = f"{tf_prefix}/moving_mean"
    mapping[f"{torch_prefix}.running_var"] = f"{tf_prefix}/moving_variance"

    return mapping


def add_bottleneck(mapping: dict, block: int, unit: int, conv: int) -> dict:
    torch_prefix = f"layer{block}.{unit-1}"
    tf_prefix = f"resnet_v1_50/block{block}/unit_{unit}/bottleneck_v1/conv{conv}"

    mapping[f"{torch_prefix}.conv{conv}.weight"] = f"{tf_prefix}/weights"

    torch_prefix = f"{torch_prefix}.bn{conv}"
    return add_batchnorm(mapping, torch_prefix, tf_prefix)


def add_shortcut(mapping: dict, block: int, unit: int) -> dict:
    torch_prefix = f"layer{block}.{unit-1}.downsample"
    tf_prefix = f"resnet_v1_50/block{block}/unit_{unit}/bottleneck_v1/shortcut"
    mapping[f"{torch_prefix}.0.weight"] = f"{tf_prefix}/weights"

    return add_batchnorm(mapping, f"{torch_prefix}.1", tf_prefix)


def add_unit(mapping: dict, block: int, unit: int) -> dict:
    add_bottleneck(mapping, block, unit, 1)
    mapping = add_bottleneck(mapping, block, unit, 2)
    mapping = add_bottleneck(mapping, block, unit, 3)

    if unit == 1:
        return add_shortcut(mapping, block, unit)
    return mapping


def add_block(mapping: dict, block: int, num_units: int) -> dict:
    for i in range(num_units):
        mapping = add_unit(mapping, block, i+1)
    return mapping


def get_default_mapping() -> dict:
    """Get the default ResNet mapping. TODO attentiton and other layers are not yet respected

    Returns:
        dict: [description]
    """
    mapping = {}

    mapping["conv1.weight"] = "resnet_v1_50/conv1/weights"
    mapping = add_batchnorm(mapping, "bn1", "resnet_v1_50/conv1")

    # This should be right, are batch-norm and conv the other way round in the two implementations?
    mapping = add_block(mapping, 1, 3)
    mapping = add_block(mapping, 2, 4)
    mapping = add_block(mapping, 3, 6)
    mapping = add_block(mapping, 4, 3)
    return mapping
