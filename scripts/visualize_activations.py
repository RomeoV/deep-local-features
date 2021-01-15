# coding: utf-8
import torch
from lib.autoencoder import FeatureEncoder1
from lib.correspondence_datamodule import CorrespondenceDataModule
dm = CorrespondenceDataModule()
dm.prepare_data()
dm.setup(stage='fit')
train_loader = dm.train_dataloader()
sample = next(iter(train_loader))

autoencoder = FeatureEncoder1.load_from_checkpoint("checkpoints/epoch=58-step=8554.ckpt").requires_grad_(False)
activations = autoencoder.get_resnet_layers(sample['image1'])

import matplotlib.pyplot as plt
act = {}
enc = {}
dec = {}
for s in autoencoder.stages:
    act[s] = activations[s]  # b x c x w x h
    enc[s] = autoencoder.encoder[s](act[s])
    dec[s] = autoencoder.decoder[s](enc[s])

def svd_compression(batch):
    svd_in = (batch.reshape(batch.shape[0], batch.shape[1], -1).permute(0, 2, 1))
    u, s, v = torch.svd(svd_in)
    compression = u[0] @ torch.diag(s[0]) @ (v[0].t()[:,:3])  # pick 3 components
    compression = compression.reshape(batch.shape[2], batch.shape[3], 3)
    return compression

def normalize_to_0_1(img):
    """ Necessary for plotting """
    return (img - img.min()) / (img.max() - img.min())

def capitalize(s):
    return s[0].upper() + s[1:]

fig = plt.figure()
for i, s in enumerate(autoencoder.stages):
    ax = plt.subplot(3,4,i+2)
    ax.imshow(normalize_to_0_1(svd_compression(act[s])))
    if i == 0:
        ax.set_ylabel("Activation")
    ax.set_title(capitalize(s))
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    ax = plt.subplot(3,4,i+6)
    ax.imshow(normalize_to_0_1(svd_compression(enc[s])))
    if i == 0:
        ax.set_ylabel("Encoded")
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    ax = plt.subplot(3,4,i+10)
    ax.imshow(normalize_to_0_1(svd_compression(dec[s])))
    if i == 0:
        ax.set_ylabel("Decoded")
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
ax = plt.subplot(3,4,5)
ax.imshow(sample['image1'][0].permute(1,2,0)/256)
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ax.set_title('Input')
fig.tight_layout()
plt.show()
