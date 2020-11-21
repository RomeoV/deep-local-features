import os
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from lib.megadepth_dataset import MegaDepthDataset

class ResNetIntermediateExtractionTransformer:
    """ Transforms MegaDepth datapoints to dict of extracted intermediate states """
    def __init__(self, net: nn.Module = None):
        if net is None:
            self.net = torchvision.models.resnet18(pretrained=True).eval()
        else:
            self.net = net.eval()

        self.activations = {}
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach().squeeze(0)
            return hook
        for l in [1,2,3,4]:
            self.net.__dict__['_modules'][f"layer{l}"][0].conv1.register_forward_hook(get_activation(f"layer{l}_conv1"))

    def __call__(self, sample):
        self.activations.clear()
        _ = self.net(sample['image1'].unsqueeze(0))
        return self.activations

def test_MegaDepthDataset_path():
    assert "MegaDepthDatasetPath" in os.environ.keys(), "Please set the environment variable 'MegaDepthDatasetPath'"

def test_load_dataset():
    assert "MegaDepthDatasetPath" in os.environ.keys(), "Please set the environment variable 'MegaDepthDatasetPath'"
    base_path = os.environ['MegaDepthDatasetPath']
    dataset = MegaDepthDataset(scene_list_path=f"{base_path}/train_scenes.txt", scene_info_path=f"{base_path}/scene_info", base_path=base_path)
    dataset.build_dataset()
    assert("image1" in dataset[0].keys())

def test_dataset_transformation():
    extraction_transformer = ResNetIntermediateExtractionTransformer()
    assert "MegaDepthDatasetPath" in os.environ.keys(), "Please set the environment variable 'MegaDepthDatasetPath'"
    base_path = os.environ['MegaDepthDatasetPath']
    dataset = MegaDepthDataset(scene_list_path=f"{base_path}/train_scenes.txt", scene_info_path=f"{base_path}/scene_info", base_path=base_path, transform=extraction_transformer)
    dataset.build_dataset()

    dl = DataLoader(dataset, batch_size=4)
    sample = next(iter(dl))

    assert("layer1_conv1" in sample)
    assert(sample["layer1_conv1"].shape[0] == 4)
