import torch
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from lib import torchvision_resnet

from torch.nn import functional as F
import pytorch_lightning

from lib.correspondence_datamodule import ResnetCorrespondenceExtractor, ResnetActivationExtractor, CorrespondenceDataModule
from lib.tf_weight_loader import load_weights
from lib.tf_weight_loader import mapping as default_mapping
from pytorch_lightning.loggers import TensorBoardLogger


def get_basic_encoder(input_channels, encoded_channels, use_relu):
    early_layers = [
        nn.Conv2d(
            in_channels=input_channels['early'], out_channels=encoded_channels, kernel_size=(1, 1))
    ]
    middle_layers = [
        nn.Conv2d(
            in_channels=input_channels['middle'], out_channels=encoded_channels, kernel_size=(1, 1))
    ]
    deep_layers = [
        nn.Conv2d(
            in_channels=input_channels['deep'], out_channels=encoded_channels, kernel_size=(1, 1))
    ]

    if use_relu:
        early_layers.append(nn.ReLU(True))
        middle_layers.append(nn.ReLU(True))
        deep_layers.append(nn.ReLU(True))

    return {
        'early': nn.Sequential(*early_layers),
        'middle': nn.Sequential(*middle_layers),
        'deep': nn.Sequential(*deep_layers)
    }


def get_basic_decoder(input_channels, encoded_channels, use_relu):
    early_layers = [
        nn.ConvTranspose2d(
            in_channels=input_channels, out_channels=encoded_channels['early'], kernel_size=(1, 1))
    ]
    middle_layers = [
        nn.ConvTranspose2d(
            in_channels=input_channels, out_channels=encoded_channels['middle'], kernel_size=(1, 1))
    ]
    deep_layers = [
        nn.ConvTranspose2d(
            in_channels=input_channels, out_channels=encoded_channels['deep'], kernel_size=(1, 1))
    ]

    if use_relu:
        early_layers.append(nn.ReLU(True))
        middle_layers.append(nn.ReLU(True))
        deep_layers.append(nn.ReLU(True))

    return {
        'early': nn.Sequential(*early_layers),
        'middle': nn.Sequential(*middle_layers),
        'deep': nn.Sequential(*deep_layers)
    }


class EncoderBase(LightningModule):
    def __init__(self, encoded_channels, input_channels, load_tf_weights=True, **resnet_args):
        super().__init__()
        self.resnet = torchvision_resnet.resnet50(
            pretrained=True, **resnet_args).eval().requires_grad_(False)

        if load_tf_weights:
            mapping = default_mapping.get_default_mapping()
            weight_loader = load_weights.WeightLoader(mapping=mapping)
            self.resnet = weight_loader.set_torch_model(self.resnet)

        self.encoded_channels = encoded_channels
        self.input_channels = input_channels
        self.stages = ('early', 'middle', 'deep')

        self.resnet_extractor = ResnetActivationExtractor(
            self.resnet, conv_layer=None)
        self.resnet_correspondence_extractor = ResnetCorrespondenceExtractor(
            self.resnet_extractor)

        self.encoder, self.decoder = self.get_encoder_decoder()

        # we need this such that the encoders get tranfered to gpu automatically
        self.e1 = self.encoder['early']
        self.e2 = self.encoder['middle']
        self.e3 = self.encoder['deep']

        self.d1 = self.decoder['early']
        self.d2 = self.decoder['middle']
        self.d3 = self.decoder['deep']

    def get_encoder_decoder(self):
        raise NotImplementedError()

    def forward(self, x):
        x = self.get_resnet_layers(x)

        y = {}
        y['early'] = self.encoder['early'](x['early'])
        y['middle'] = self.encoder['middle'](x['middle'])
        y['deep'] = self.encoder['deep'](x['deep'])

        return y

    def configure_optimizers(self):
        raise NotImplementedError()

    def get_resnet_layers(self, x):
        raise NotImplementedError()


class FeatureEncoderUp(EncoderBase):
    def __init__(self, encoded_channels=64, load_tf_weights=True, **resnet_args):
        super().__init__(encoded_channels, {
            'early': 512,
            'middle': 1024,
            'deep': 2048,
        }, load_tf_weights=load_tf_weights, **resnet_args)

    def get_encoder_decoder(self):
        return (get_basic_encoder(self.input_channels, self.encoded_channels, use_relu=False),
                get_basic_decoder(self.encoded_channels, self.input_channels, use_relu=True))

    @torch.no_grad()
    def get_resnet_layers(self, x):
        activations = self.resnet_extractor(x)
        activation_transform = {
            'early': lambda x: x,
            'middle': nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            'deep': nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
        }
        return {'early': activation_transform['early'](activations["layer2"]),
                'middle': activation_transform['middle'](activations["layer3"]),
                'deep': activation_transform['deep'](activations["layer4"])}

    def training_step(self, batch, batch_idx):
        x1 = self.get_resnet_layers(batch["image1"])
        x2 = self.get_resnet_layers(batch["image2"])

        z1 = {}
        x_hat1 = {}
        z2 = {}
        x_hat2 = {}
        for s in self.stages:
            z1[s] = self.encoder[s](x1[s])
            x_hat1[s] = self.decoder[s](z1[s])
            z2[s] = self.encoder[s](x2[s])
            x_hat2[s] = self.decoder[s](z2[s])

        loss = sum((F.mse_loss(x1[s], x_hat1[s]) +
                    F.mse_loss(x2[s], x_hat2[s])) for s in self.stages)

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x1 = self.get_resnet_layers(batch["image1"])
        x2 = self.get_resnet_layers(batch["image2"])

        z1 = {}
        x_hat1 = {}
        z2 = {}
        x_hat2 = {}
        for s in self.stages:
            z1[s] = self.encoder[s](x1[s])
            x_hat1[s] = self.decoder[s](z1[s])
            z2[s] = self.encoder[s](x2[s])
            x_hat2[s] = self.decoder[s](z2[s])

        loss = sum((F.mse_loss(x1[s], x_hat1[s]) +
                    F.mse_loss(x2[s], x_hat2[s])) for s in self.stages)

        self.log('validation_loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1.0*1e-3)
        return optimizer


class FeatureEncoder32Up(FeatureEncoderUp):
    def __init__(self, load_tf_weights=True, **resnet_args):
        super().__init__(encoded_channels=32, load_tf_weights=load_tf_weights, **resnet_args)


class FeatureEncoder64Up(FeatureEncoderUp):
    def __init__(self, load_tf_weights=True, **resnet_args):
        super().__init__(encoded_channels=64, load_tf_weights=load_tf_weights, **resnet_args)


class FeatureEncoder(EncoderBase):
    def __init__(self, encoded_channels=32, load_tf_weights=True, **resnet_args):
        super().__init__(encoded_channels, {
            'early': 512,
            'middle': 1024,
            'deep': 2048,
        }, load_tf_weights=load_tf_weights, **resnet_args)

        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.input_channels['deep'],
                               out_channels=self.input_channels['deep'], kernel_size=(1, 1), stride=2),
            nn.ReLU(True)
        )

        self.downsampler = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels['early'],
                      out_channels=self.input_channels['early'], kernel_size=(1, 1), stride=2),
            nn.ReLU(True)
        )

    def get_encoder_decoder(self):
        return (get_basic_encoder(self.input_channels, self.encoded_channels, True),
                get_basic_decoder(self.encoded_channels, self.input_channels, False))

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
    def get_resnet_layers(self, x):
        activations = self.resnet_extractor(x)

        activation_transform = {
            'early': self.downsampler,
            'middle': lambda x: x,
            'deep': self.upsampler,
        }
        return {'early': activation_transform['early'](activations["layer2_conv3"]),
                'middle': activation_transform['middle'](activations["layer3_conv3"]),
                'deep': activation_transform['deep'](activations["layer4_conv3"])}


class FeatureEncoder3(EncoderBase):
    def __init__(self, encoded_channels=32, load_tf_weights=True, **resnet_args):
        super().__init__(encoded_channels, {
            'early': 512,
            'middle': 1024,
            'deep': 2048,
        }, load_tf_weights=load_tf_weights, **resnet_args)

    def get_encoder_decoder(self):
        return (get_basic_encoder(self.input_channels, self.encoded_channels, True),
                get_basic_decoder(self.encoded_channels, self.input_channels, False))

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
    def get_resnet_layers(self, x):
        activations = self.resnet_extractor(x)
        activation_transform = {
            'early': nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            'middle': lambda x: x,
            'deep': nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        }
        return {'early': activation_transform['early'](activations["layer2_conv3"]),
                'middle': activation_transform['middle'](activations["layer3_conv3"]),
                'deep': activation_transform['deep'](activations["layer4_conv3"])}


class FeatureEncoder1(EncoderBase):
    def __init__(self, encoded_channels=16, load_tf_weights=True, **resnet_args):
        super().__init__(encoded_channels, {
            'early': 128,
            'middle': 256,
            'deep': 512,
        }, load_tf_weights=load_tf_weights, **resnet_args)

    def get_encoder_decoder(self):
        return (get_basic_encoder(self.input_channels, self.encoded_channels, True),
                get_basic_decoder(self.encoded_channels, self.input_channels, True))

    def training_step(self, batch, batch_idx):
        x = self.get_resnet_layers_no_grad(batch["image1"])

        z = {}
        x_hat = {}
        for s in self.stages:
            z[s] = self.encoder[s](x[s])
            x_hat[s] = self.decoder[s](z[s])

        loss = sum((F.mse_loss(x[s], x_hat[s]) for s in self.stages))

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x = self.get_resnet_layers_no_grad(batch["image1"])

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
    def get_resnet_layers_no_grad(self, x):
        return self.get_resnet_layers(x)

    def get_resnet_layers(self, x):
        activations = self.resnet_extractor(x)
        if len(activations['layer4_conv1'].size()) == 3:
            deep_features = torch.unsqueeze(activations['layer4_conv1'], 0)
            middle_features = torch.unsqueeze(activations['layer3_conv1'], 0)
            early_features = torch.unsqueeze(activations['layer2_conv1'], 0)
        else:
            deep_features = activations['layer4_conv1']
            middle_features = activations['layer3_conv1']
            early_features = activations['layer2_conv1']
        activation_transform = {
            'early': nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            'middle': lambda x: x,
            'deep': nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        }
        return {'early': activation_transform['early'](early_features),
                'middle': activation_transform['middle'](middle_features),
                'deep': activation_transform['deep'](deep_features)}


if __name__ == "__main__":
    autoencoder = FeatureEncoderUp(encoded_channels=32)
    tb_logger = TensorBoardLogger(
        'tb_logs', name='feature_encoder32_up_deep_lr2e4')
    trainer = pytorch_lightning.Trainer(logger=tb_logger,
                                        gpus=1 if torch.cuda.is_available() else None)
    dm = CorrespondenceDataModule()
    trainer.fit(autoencoder, dm)
