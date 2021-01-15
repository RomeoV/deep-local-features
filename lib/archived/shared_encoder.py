import torch
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
import torchvision.models

from torch.nn import functional as F
import pytorch_lightning

from lib.correspondence_datamodule import ResnetCorrespondenceExtractor, ResnetActivationExtractor, \
    CorrespondenceDataModule
from lib.tf_weight_loader import load_weights
from lib.tf_weight_loader import mapping as default_mapping
from pytorch_lightning.loggers import TensorBoardLogger
from lib.repeatability_loss import RepeatabilityLoss
from lib.loss import *

REP_LOSS = False


class SharedCorrespondenceEncoder(LightningModule):
    def __init__(self, load_tf_weights=True):
        super().__init__()

        self.resnet = torchvision.models.resnet50(
            pretrained=True).eval().requires_grad_(False)

        if load_tf_weights:
            mapping = default_mapping.get_default_mapping()
            weight_loader = load_weights.WeightLoader(mapping=mapping)
            self.resnet = weight_loader.set_torch_model(self.resnet)

        self.encoded_channels = 64

        self.stages = ('early', 'middle', 'deep')

        self.pre_autoenc_channels = 512
        self.input_channels = {
            'early': 512,
            'middle': 1024,
            'deep': 2048,
        }

        self.resnet_extractor = ResnetActivationExtractor(self.resnet, conv_layer=None)
        self.resnet_correspondence_extractor = ResnetCorrespondenceExtractor(
            self.resnet_extractor)

        

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.pre_autoenc_channels, out_channels=self.encoded_channels, kernel_size=(1, 1)),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.encoded_channels,
                            out_channels=self.pre_autoenc_channels, kernel_size=(1, 1)),
            nn.ReLU(True),
        )

        self.pre_encoding = {
            'early': nn.Sequential(
                nn.Conv2d(in_channels=self.input_channels['early'], out_channels=self.pre_autoenc_channels, kernel_size=(1,1)),
                nn.Conv2d(in_channels=self.pre_autoenc_channels, out_channels=self.pre_autoenc_channels, kernel_size=(1,1))
            ),
            'middle': nn.Sequential(
                nn.Conv2d(in_channels=self.input_channels['middle'], out_channels=self.pre_autoenc_channels, kernel_size=(1,1)),
                nn.Conv2d(in_channels=self.pre_autoenc_channels, out_channels=self.pre_autoenc_channels, kernel_size=(2,2))
            ),
            'deep': nn.Sequential(
                nn.Conv2d(in_channels=self.input_channels['deep'], out_channels=self.pre_autoenc_channels, kernel_size=(1,1)),
                nn.Conv2d(in_channels=self.pre_autoenc_channels, out_channels=self.pre_autoenc_channels, kernel_size=(2,2))
            )
        }

        # we need this such that the encoders get tranfered to gpu automatically
        self.p1 = self.pre_encoding['early']
        self.p2 = self.pre_encoding['middle']
        self.p3 = self.pre_encoding['deep']

        self.correspondence_loss = CorrespondenceLoss()

        self.cfactor = 1.0

    def forward(self, x):
        x = self.get_resnet_layers(x)

        early = x['early']
        early = self.pre_encoding['early'](early)

        middle = x['middle']
        middle = self.pre_encoding['middle'](middle)

        deep = x['deep']
        deep = self.pre_encoding['deep'](deep)

        y = {}
        y['early'] = self.encoder(early)
        y['middle'] = self.encoder(middle)
        y['deep'] = self.encoder(deep)

        return y

    def training_step(self, batch, batch_idx):
        x1 = self.get_resnet_layers(batch["image1"])
        x2 = self.get_resnet_layers(batch["image2"])

        z1 = {}
        x_hat1 = {}
        x_pre1 = {}
        z2 = {}
        x_hat2 = {}
        x_pre2 = {}
        for s in self.stages:
            x_pre1[s] = self.pre_encoding[s](x1[s])
            z1[s] = self.encoder(x_pre1[s])
            x_hat1[s] = self.decoder(z1[s])

            x_pre2[s] = self.pre_encoding[s](x2[s])
            z2[s] = self.encoder(x_pre2[s])
            x_hat2[s] = self.decoder(z2[s])

        loss = sum((F.mse_loss(x_pre1[s], x_hat1[s]) + F.mse_loss(x_pre2[s],
                                                              x_hat2[s]) + self.cfactor * self.correspondence_loss(
            z1[s], z2[s], batch)) for s in self.stages)

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x1 = self.get_resnet_layers(batch["image1"])
        x2 = self.get_resnet_layers(batch["image2"])

        z1 = {}
        x_hat1 = {}
        x_pre1 = {}
        z2 = {}
        x_hat2 = {}
        x_pre2 = {}
        for s in self.stages:
            x_pre1[s] = self.pre_encoding[s](x1[s])
            z1[s] = self.encoder(x_pre1[s])
            x_hat1[s] = self.decoder(z1[s])

            x_pre2[s] = self.pre_encoding[s](x2[s])
            z2[s] = self.encoder(x_pre2[s])
            x_hat2[s] = self.decoder(z2[s])

        loss = sum((F.mse_loss(x_pre1[s], x_hat1[s]) + F.mse_loss(x_pre2[s],
                                                              x_hat2[s]) + self.cfactor * self.correspondence_loss(
            z1[s], z2[s], batch)) for s in self.stages)

        self.log('validation_loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1.0 * 1e-3)
        return optimizer

    @torch.no_grad()
    def get_resnet_layers(self, x):
        activations = self.resnet_extractor(x)
        # activation_transform = {
        #     'early': nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
        #     'middle': lambda x: x,
        #     'deep': nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        # }
        activation_transform = {
            'early': lambda x: x,
            'middle': nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            'deep': nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
        }
        return {'early': activation_transform['early'](activations["layer2"]),
                'middle': activation_transform['middle'](activations["layer3"]),
                'deep': activation_transform['deep'](activations["layer4"])}

class SharedMultiAttentionLayer2(LightningModule):
    def __init__(self, feature_encoder):
        super().__init__()
        self.feature_encoder = feature_encoder
        if (REP_LOSS):
            self.loss = RepeatabilityLoss()
        else:
            self.loss = DistinctivenessLoss()
        self.stages = ('early', 'middle', 'deep')
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_encoder.encoded_channels,
                      out_channels=512, kernel_size=(1, 1)),  # bx2xWxH
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, \
                      out_channels=1, kernel_size=(1, 1)),  # bx2xWxH
        )

    def softmax(self, ux):
        if ux.shape[1] == 1:
            x = F.softplus(ux)
            # for sure in [0,1], much less plateaus than softmax
            return x / (1 + x)
        elif ux.shape[1] == 2:
            return F.softmax(ux, dim=1)[:, 1:2]

    def forward(self, x):
        y = {}
        y["early"] = self.softmax(self.attention(x["early"]))
        y["middle"] = self.softmax(self.attention(x["middle"]))
        y["deep"] = self.softmax(self.attention(x["deep"]))
        return y

    def training_step(self, batch, batch_idx):
        x1 = self.feature_encoder.get_resnet_layers(batch["image1"])
        x2 = self.feature_encoder.get_resnet_layers(batch["image2"])

        z1 = {}
        x_hat1 = {}
        x_pre1 = {}
        z2 = {}
        x_hat2 = {}
        x_pre2 = {}
        for s in self.stages:
            x_pre1[s] = self.feature_encoder.pre_encoding[s](x1[s])
            z1[s] = self.feature_encoder.encoder(x_pre1[s])
            x_hat1[s] = self.feature_encoder.decoder(z1[s])

            x_pre2[s] = self.feature_encoder.pre_encoding[s](x2[s])
            z2[s] = self.feature_encoder.encoder(x_pre2[s])
            x_hat2[s] = self.feature_encoder.decoder(z2[s])

        loss = sum((F.mse_loss(x_pre1[s], x_hat1[s]) + F.mse_loss(x_pre2[s], x_hat2[s]) + self.feature_encoder.cfactor * self.feature_encoder.correspondence_loss(z1[s],z2[s], batch)) for s in x1.keys())
    
        y1 = self.forward(z1)
        y2 = self.forward(z2)

        for layer in x1.keys():
            loss = loss +  self.loss(z1[layer], z2[layer], y1[layer], y2[layer], batch)
        
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x1 = self.feature_encoder.get_resnet_layers(batch["image1"])
        x2 = self.feature_encoder.get_resnet_layers(batch["image2"])

        z1 = {}
        x_hat1 = {}
        x_pre1 = {}
        z2 = {}
        x_hat2 = {}
        x_pre2 = {}
        for s in self.stages:
            x_pre1[s] = self.feature_encoder.pre_encoding[s](x1[s])
            z1[s] = self.feature_encoder.encoder(x_pre1[s])
            x_hat1[s] = self.feature_encoder.decoder(z1[s])

            x_pre2[s] = self.feature_encoder.pre_encoding[s](x2[s])
            z2[s] = self.feature_encoder.encoder(x_pre2[s])
            x_hat2[s] = self.feature_encoder.decoder(z2[s])

        loss = sum((F.mse_loss(x_pre1[s], x_hat1[s]) + F.mse_loss(x_pre2[s], x_hat2[s]) + self.feature_encoder.cfactor * self.feature_encoder.correspondence_loss(z1[s],z2[s], batch)) for s in x1.keys())
    
        y1 = self.forward(z1)
        y2 = self.forward(z2)

        for layer in x1.keys():
            loss = loss +  self.loss(z1[layer], z2[layer], y1[layer], y2[layer], batch)
        
        self.log('validation_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5*1e-4)
        return optimizer


if __name__ == "__main__":
    autoencoder = SharedCorrespondenceEncoder()
    attention = SharedMultiAttentionLayer2(autoencoder)
    tb_logger = TensorBoardLogger('tb_logs', name='shared_weight_model')
    trainer = pytorch_lightning.Trainer(logger=tb_logger,
                                        gpus=1 if torch.cuda.is_available() else None)
    dm = CorrespondenceDataModule()
    trainer.fit(attention, dm)
    # attention = SharedMultiAttentionLayer2(autoencoder)
    # i = torch.normal(0,1,(1,3,640,480))
    # batch = {
    #     "image1": i, 
    #     "image2": i
    # }

    # attention.training_step(batch, 0)