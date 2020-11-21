import os
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
from megadepth_dataset import MegaDepthDataset
import pytorch_lightning


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


class AutoencoderDataModule(pytorch_lightning.LightningDataModule):
    """ Autoencoder Data Module for pytorch_lightning

    Loads MegaDepth dataset (train, val, test) and automatically extracts the
    internal representations.
    
    Can be used very easily with pytorch_lightning training architecture.
    """

    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
    
    def prepare_data(self):
        """ Builds MegaDepthDataset(s) with extraction """
        print("Running data module setup!")
        assert "MegaDepthDatasetPath" in os.environ.keys(), "Please set the environment variable 'MegaDepthDatasetPath'"
        base_path = os.environ['MegaDepthDatasetPath']

        extraction_transformer = ResNetIntermediateExtractionTransformer()
        self.dataset_train = MegaDepthDataset(train=True, scene_list_path=f"{base_path}/train_scenes.txt", scene_info_path=f"{base_path}/scene_info", base_path=base_path, transform=extraction_transformer)
        self.dataset_train.build_dataset()

        self.dataset_test = MegaDepthDataset(train=False, scene_list_path=f"{base_path}/train_scenes.txt", scene_info_path=f"{base_path}/scene_info", base_path=base_path, transform=extraction_transformer)
        self.dataset_test.build_dataset()

    def setup(self, stage):
        N = len(self.dataset_train)
        if stage == 'fit':
            self.autoencoder_train, self.autoencoder_val = random_split(self.dataset_train, [round(N*0.8), round(N*0.2)])
        elif stage == 'test':
            self.autoencoder_test = self.dataset_test

    def train_dataloader(self):
        autoencoder_train = DataLoader(self.autoencoder_train, batch_size=self.batch_size)
        return autoencoder_train

    def val_dataloader(self):
        autoencoder_val = DataLoader(self.autoencoder_val, batch_size=self.batch_size)
        return autoencoder_val

    def test_dataloader(self):
        autoencoder_test = DataLoader(self.autoencoder_test, batch_size=self.batch_size)
        return autoencoder_test


def test_AEDataModule():
    # stage = 'fit'
    data_module = AutoencoderDataModule(batch_size=8)
    data_module.prepare_data()
    data_module.setup(stage='fit')
    dl_train = data_module.train_dataloader()
    dl_val = data_module.val_dataloader()

    sample_train = next(iter(dl_train))
    assert("layer1_conv1" in sample_train)
    assert(sample_train["layer1_conv1"].shape[0] == 8)
    sample_val = next(iter(dl_val))
    assert("layer1_conv1" in sample_val)
    assert(sample_val["layer1_conv1"].shape[0] == 8)

    # stage = 'test'
    data_module = AutoencoderDataModule(batch_size=8)
    data_module.prepare_data()
    data_module.setup(stage='test')
    dl_test = data_module.test_dataloader()

    sample_test = next(iter(dl_test))
    assert("layer1_conv1" in sample_test)
    assert(sample_test["layer1_conv1"].shape[0] == 8)
