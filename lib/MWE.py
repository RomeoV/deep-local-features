import torch
import torchvision.models
from lib.megadepth_dataset import MegaDepthDataset
from PIL import Image

base_path = "/home/romeo/Documents/ETH/polybox/Shared/Deep Learning/MegaDepthDataset"
dataset = MegaDepthDataset(scene_list_path=f"{base_path}/train_scenes.txt", scene_info_path=f"{base_path}/scene_info", base_path=base_path)
dataset.build_dataset()

first_datapoint = dataset[0]
#repeated_image = torch.stack([first_datapoint['image1'] for _ in range(64)], dim=0)  # to fulfill batch_size = 64

if False:  # plot image1
    Image.fromarray(np.sum(d['image1'].numpy().transpose(1,2,0), axis=2)/3).show()

resnet18 = torchvision.models.resnet18(pretrained=True).eval()

activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

for l in [1,2,3,4]:
    resnet18.__dict__['_modules'][f"layer{l}"][0].conv1.register_forward_hook(get_activation(f"layer{l}_conv1"))

logits = resnet18(first_datapoint['image1'].unsqueeze(0))
