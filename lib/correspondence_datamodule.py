import os
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
import pytorch_lightning
import multiprocessing
from lib.megadepth_dataset import MegaDepthDataset

class ResnetActivationExtractor:
    """ Transforms MegaDepth datapoints to dict of extracted intermediate states """
    def __init__(self, net: nn.Module, conv_layer = 3):
        self.net = net.eval()
        self.activations = {}
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach() # .squeeze(0)
            return hook
        for l in [1,2,3,4]:
            if (conv_layer is None):
                self.net.__dict__['_modules'][f"layer{l}"].register_forward_hook(get_activation(f"layer{l}"))
            if (conv_layer == 3):
                self.net.__dict__['_modules'][f"layer{l}"][0].conv3.register_forward_hook(get_activation(f"layer{l}_conv3"))
            if (conv_layer == 1):
                self.net.__dict__['_modules'][f"layer{l}"][0].conv1.register_forward_hook(get_activation(f"layer{l}_conv1"))

    def __call__(self, image):
        self.activations.clear()
        _ = self.net(image)
        return self.activations

class ResnetCorrespondenceExtractor:
    """ Transforms MegaDepth datapoints to dict of extracted intermediate states """
    def __init__(self, resnet_extractor):
        self.resnet_extractor = resnet_extractor

    def __call__(self, sample):
        result = {}
        result["activations1"] = self.resnet_extractor(sample["image1"])
        result["activations2"] = self.resnet_extractor(sample["image2"])
        return result

# def test_MegaDepthDataset_path():
#     assert "MegaDepthDatasetPath" in os.environ.keys(), "Please set the environment variable 'MegaDepthDatasetPath'"

# def test_load_dataset():
#     assert "MegaDepthDatasetPath" in os.environ.keys(), "Please set the environment variable 'MegaDepthDatasetPath'"
#     base_path = os.environ['MegaDepthDatasetPath']
#     dataset = MegaDepthDataset(scene_list_path=f"{base_path}/train_scenes.txt", scene_info_path=f"{base_path}/scene_info", base_path=base_path)
#     dataset.build_dataset()
#     assert("image11" in dataset[0].keys())

# def test_dataset_transformation():
#     extraction_transformer = ResnetCorrespondenceExtractor()
#     assert "MegaDepthDatasetPath" in os.environ.keys(), "Please set the environment variable 'MegaDepthDatasetPath'"
#     base_path = os.environ['MegaDepthDatasetPath']
#     dataset = MegaDepthDataset(scene_list_path=f"{base_path}/train_scenes.txt", scene_info_path=f"{base_path}/scene_info", base_path=base_path, transform=extraction_transformer)
#     dataset.build_dataset()

#     dl = DataLoader(dataset, batch_size=4)
#     sample = next(iter(dl))

#     assert("layer1_conv1" in sample["activations1"])
#     assert(sample["activations1"]["layer1_conv1"].shape[0] == 4)


class CorrespondenceDataModule(pytorch_lightning.LightningDataModule):
    """ Correspondence Data Module for pytorch_lightning

    Loads MegaDepth dataset (train, val, test) and automatically extracts the
    internal representations.
    
    Can be used very easily with pytorch_lightning training architecture.
    """

    def __init__(self, base_path=None, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = (multiprocessing.cpu_count() - 2) if multiprocessing.cpu_count() >= 4 else 1
        if base_path is None:
            self.base_path = os.environ['MegaDepthDatasetPath']
        else:
            self.base_path = base_path
    
    def prepare_data(self):
        """ Builds MegaDepthDataset(s) with extraction """
        base_path = self.base_path
        self.dataset_train = MegaDepthDataset(train=True, scene_list_path=f"{base_path}/train_scenes.txt", scene_info_path=f"{base_path}/scene_info", base_path=base_path)
        self.dataset_train.build_dataset()

        self.dataset_test = MegaDepthDataset(train=False, scene_list_path=f"{base_path}/valid_scenes.txt", scene_info_path=f"{base_path}/scene_info", base_path=base_path)
        self.dataset_test.build_dataset()

    def setup(self, stage):
        N = len(self.dataset_train)
        if stage == 'fit':
            self.correspondence_train, self.correspondence_val = random_split(self.dataset_train, [round(N*0.8), round(N*0.2)])
        elif stage == 'test':
            self.correspondence_test = self.dataset_test

    def train_dataloader(self):
        correspondence_train = DataLoader(self.correspondence_train, batch_size=self.batch_size, num_workers=self.num_workers)
        return correspondence_train

    def val_dataloader(self):
        correspondence_val = DataLoader(self.correspondence_val, batch_size=self.batch_size, num_workers=self.num_workers)
        return correspondence_val

    def test_dataloader(self):
        correspondence_test = DataLoader(self.correspondence_test, batch_size=self.batch_size, num_workers=self.num_workers)
        return correspondence_test


# def test_AttentionDataModule():
#     # stage = 'fit'
#     data_module = AttentionDataModule(batch_size=8)
#     data_module.prepare_data()
#     data_module.setup(stage='fit')
#     dl_train = data_module.train_dataloader()
#     dl_val = data_module.val_dataloader()

#     sample_train = next(iter(dl_train))
#     # assert("layer1_conv1" in sample_train)
#     assert(sample_train["layer1_conv1"].shape[0] == 8)
#     sample_val = next(iter(dl_val))
#     # assert("layer1_conv1" in sample_val)
#     # assert(sample_val["layer1_conv1"].shape[0] == 8)

#     # stage = 'test'
#     data_module = AutoencoderDataModule(batch_size=8)
#     data_module.prepare_data()
#     data_module.setup(stage='test')
#     dl_test = data_module.test_dataloader()

#     sample_test = next(iter(dl_test))
#     # assert("layer1_conv1" in sample_test)
#     # assert(sample_test["layer1_conv1"].shape[0] == 8)