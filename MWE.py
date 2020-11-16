import torch
import torchvision.models
from lib.megadepth_dataset import MegaDepthDataset
from PIL import Image

base_path = "/home/romeo/Documents/ETH/polybox/Shared/Deep Learning/MegaDepthDataset"
dataset = MegaDepthDataset(scene_list_path=f"{base_path}/train_scenes.txt", scene_info_path=f"{base_path}/scene_info", base_path=base_path)
dataset.build_dataset()

first_datapoint = dataset[0]
repeated_image = torch.stack([first_datapoint['image1'] for _ in range(64)], dim=0)  # to fulfill batch_size = 64

if False:  # plot image1
    Image.fromarray(np.sum(d['image1'].numpy().transpose(1,2,0), axis=2)/3).show()

resnet18 = torchvision.models.resnet18(pretrained=True)
logits = resnet18(repeated_image)
print(logits[0])
