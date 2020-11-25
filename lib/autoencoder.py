import torch
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
import torchvision.models

from torch.nn import functional as F
import pytorch_lightning

from lib.correspondence_datamodule import ResnetCorrespondenceExtractor, ResnetActivationExtractor

class FeatureEncoder(LightningModule):
    def __init__(self):
        super().__init__()

        self.resnet = torchvision.models.resnet50(pretrained=True)

        self.encoded_channels = 16

        self.early_input_channels = 128
        self.middle_input_channels = 256
        self.deep_input_channels = 512

        self.resnet_extractor = ResnetActivationExtractor(self.resnet)
        self.resnet_correspondence_extractor = ResnetCorrespondenceExtractor(self.resnet_extractor)

        self.encoder_early = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2,2), stride = (2,2)),
            nn.Conv2d(in_channels=self.early_input_channels, out_channels=self.encoded_channels, kernel_size=(1,1)), 
            nn.ReLU(True),
        )
        self.decoder_early = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.encoded_channels, out_channels=self.early_input_channels, kernel_size=(1,1)), 
            nn.Sigmoid(),
        )

        self.encoder_middle = nn.Sequential(
            nn.Conv2d(in_channels=self.middle_input_channels, out_channels=self.encoded_channels, kernel_size=(1,1)), 
            nn.ReLU(True),
        )
        self.decoder_middle = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.encoded_channels, out_channels=self.middle_input_channels, kernel_size=(1,1)), 
            nn.Sigmoid(),
        )

        self.encoder_deep = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels=self.deep_input_channels, out_channels=self.encoded_channels, kernel_size=(1,1)), 
            nn.ReLU(True),
        )
        self.decoder_deep = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.encoded_channels, out_channels=self.deep_input_channels, kernel_size=(1,1)), 
            nn.Sigmoid(),
        )

    def forward(self, x):
        (x_early, x_middle, x_deep) = self.get_resnet_layers(x)

        y_early = self.encoder_early(x_early)
        y_middle = self.encoder_middle(x_middle)
        y_deep = self.encoder_deep(x_deep)

        return [y_early, y_middle, y_deep]

    def training_step(self, batch, batch_idx):
        (x_early, x_middle, x_deep) = get_resnet_layers(batch["image1"])

        z_early = self.encoder_early(x_early)
        z_middle = self.encoder_middle(x_middle)
        z_deep = self.encoder_deep(x_deep)

        x_hat_early = self.decoder_early(z_early)
        x_hat_middle = self.decoder_middle(z_middle)
        x_hat_deep = self.decoder_deep(z_deep)

        loss = F.mse_loss(x_early, x_hat_early) + F.mse_loss(x_middle, x_hat_middle) + F.mse_loss(x_deep, x_hat_deep)

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        (x_early, x_middle, x_deep) = get_resnet_layers(batch["image1"])

        z_early = self.encoder_early(x_early)
        z_middle = self.encoder_middle(x_middle)
        z_deep = self.encoder_deep(x_deep)

        x_hat_early = self.decoder_early(z_early)
        x_hat_middle = self.decoder_middle(z_middle)
        x_hat_deep = self.decoder_deep(z_deep)

        loss = F.mse_loss(x_early, x_hat_early) + F.mse_loss(x_middle, x_hat_middle) + F.mse_loss(x_deep, x_hat_deep)

        self.log('validation_loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    @torch.no_grad()
    def get_resnet_layers(self,x):
        activations = self.resnet_extractor(x)
        # print(activations.keys())
        return (activations["layer2_conv1"], activations["layer3_conv1"], activations["layer4_conv1"])

if __name__ == "__main__":
    autoencoder = FeatureEncoder()
    trainer = pytorch_lightning.Trainer(gpus=1 if torch.cuda.is_available() else None)
    dm = CorrespondenceDataModule()
    trainer.fit(autoencoder, dm)
