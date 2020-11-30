import torch
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision.models

from torch.nn import functional as F
import pytorch_lightning

from lib.correspondence_datamodule import ResnetCorrespondenceExtractor, ResnetActivationExtractor, CorrespondenceDataModule

class FeatureEncoder(LightningModule):
    def __init__(self):
        super().__init__()

        self.resnet = torchvision.models.resnet50(pretrained=True)

        self.encoded_channels = 16

        self.stages = ('early', 'middle', 'deep')

        self.input_channels = {
            'early': 128,
            'middle': 256,
            'deep': 512,
        }

        self.resnet_extractor = ResnetActivationExtractor(self.resnet)
        self.resnet_correspondence_extractor = ResnetCorrespondenceExtractor(self.resnet_extractor)

        self.encoder = {
            'early': nn.Sequential(
                nn.Conv2d(in_channels=self.input_channels['early'], out_channels=self.encoded_channels, kernel_size=(1,1)), 
                nn.ReLU(True),
                ),
            'middle': nn.Sequential(
                nn.Conv2d(in_channels=self.input_channels['middle'], out_channels=self.encoded_channels, kernel_size=(1,1)), 
                nn.ReLU(True),
                ),
            'deep': nn.Sequential(
                nn.Conv2d(in_channels=self.input_channels['deep'], out_channels=self.encoded_channels, kernel_size=(1,1)), 
                nn.ReLU(True),
                ),
        }

        self.decoder = {
            'early': nn.Sequential(
                nn.ConvTranspose2d(in_channels=self.encoded_channels, out_channels=self.input_channels['early'], kernel_size=(1,1)), 
                #nn.Sigmoid(),
                ),
            'middle': nn.Sequential(
                nn.ConvTranspose2d(in_channels=self.encoded_channels, out_channels=self.input_channels['middle'], kernel_size=(1,1)), 
                #nn.Sigmoid(),
                ),
            'deep': nn.Sequential(
                nn.ConvTranspose2d(in_channels=self.encoded_channels, out_channels=self.input_channels['deep'], kernel_size=(1,1)), 
                #nn.Sigmoid(),
                ),
        }

    def forward(self, x):
        x = self.get_resnet_layers(x)

        y = {}
        y['early'] = self.encoder['early'](x['early'])
        y['middle'] = self.encoder['middle'](x['middle'])
        y['deep'] = self.encoder['deep'](x['deep'])

        return y

    def training_step(self, batch, batch_idx):
        x = self.get_resnet_layers(batch["image1"])

        z = {}
        x_hat = {}
        for s in self.stages:
            z[s] = self.encoder[s](x[s])
            x_hat[s] = self.decoder[s](z[s])

        loss = sum((F.mse_loss(x[s], x_hat[s]) for s in self.stages))

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x = self.get_resnet_layers(batch["image1"])

        z = {}
        x_hat = {}
        for s in self.stages:
            z[s] = self.encoder[s](x[s])
            x_hat[s] = self.decoder[s](z[s])

        loss = sum((F.mse_loss(x[s], x_hat[s]) for s in self.stages))

        self.log('validation_loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    @torch.no_grad()
    def get_resnet_layers(self,x):
        activations = self.resnet_extractor(x)
        activation_transform = {
            'early': nn.AvgPool2d(kernel_size=(2,2), stride = (2,2)),
            'middle': lambda x: x,
            'deep': nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        }
        return {'early': activation_transform['early'](activations["layer2_conv1"]),
                'middle': activation_transform['middle'](activations["layer3_conv1"]),
                'deep': activation_transform['deep'](activations["layer4_conv1"])}

if __name__ == "__main__":
    autoencoder = FeatureEncoder()
    tb_logger = TensorBoardLogger('tb_logs', name='autoencoder')
    trainer = pytorch_lightning.Trainer(logger=tb_logger, gpus=1 if torch.cuda.is_available() else None)
    dm = CorrespondenceDataModule()
    trainer.fit(autoencoder, dm)
