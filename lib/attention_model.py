import torch
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger

from torch.nn import functional as F
import pytorch_lightning

from lib.loss import *
from lib.repeatability_loss import RepeatabilityLoss
from lib.autoencoder import *

REP_LOSS = False

class AttentionLayer(LightningModule):
    def __init__(self, feature_encoder):
        super().__init__()
        self.feature_encoder = feature_encoder
        self.loss = DistinctivenessLoss()

        self.attentions = nn.Conv2d(in_channels=self.feature_encoder.encoded_channels, \
                    out_channels=2, kernel_size=(1,1)) #bx2xWxH

    def softmax(self, ux):
        if ux.shape[1] == 1:
            x = F.softplus(ux)
            return x / (1 + x)  # for sure in [0,1], much less plateaus than softmax
        elif ux.shape[1] == 2:
            return F.softmax(ux, dim=1)[:,1:2]
    
    def forward(self, x):
        x = self.attentions(x)
        x = self.softmax(x) #bx1xWxH
        return x
    
    def training_step(self, batch, batch_idx):
        x1 = batch['image1']
        x2 = batch["image2"]

        with torch.no_grad():
            x1_encoded = self.sum_layers(self.feature_encoder.forward(x1))
            x2_encoded = self.sum_layers(self.feature_encoder.forward(x2))

        # x1_encoded.requires_grad = False
        # x2_encoded.requires_grad = False
        y1 = self.forward(x1_encoded)
        y2 = self.forward(x2_encoded)

        loss = self.loss(x1_encoded, x2_encoded, y1, y2, batch)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x1 = batch['image1']
        x2 = batch["image2"]

        with torch.no_grad():
            x1_encoded = self.sum_layers(self.feature_encoder.forward(x1))
            x2_encoded = self.sum_layers(self.feature_encoder.forward(x2))

        # x1_encoded.requires_grad = False
        # x2_encoded.requires_grad = False
        y1 = self.forward(x1_encoded)
        y2 = self.forward(x2_encoded)

        loss = self.loss(x1_encoded, x2_encoded, y1, y2, batch)

        self.log('validation_loss', loss)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def concat_layers(self, x_dict):
        return torch.cat([x_dict["early"], x_dict["middle"], x_dict["deep"]], 1) #bx48xWxH

    def sum_layers(self, x_dict):
        return x_dict["early"] + x_dict["middle"] + x_dict["deep"] #bx48xWxH


class AttentionLayer2(LightningModule):
    def __init__(self, feature_encoder):
        super().__init__()
        self.feature_encoder = feature_encoder
        self.loss = DistinctivenessLoss()

        self.conv1 = nn.Conv2d(in_channels=self.feature_encoder.encoded_channels, \
                    out_channels=16, kernel_size=(1,1)) #bx2xWxH
        self.activation = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, \
                    out_channels=1, kernel_size=(1,1)) #bx2xWxH
        
        self.agg = nn.Softplus(beta=1, threshold=20)

    def softmax(self, ux):
        if ux.shape[1] == 1:
            x = F.softplus(ux)
            return x / (1 + x)  # for sure in [0,1], much less plateaus than softmax
        elif ux.shape[1] == 2:
            return F.softmax(ux, dim=1)[:,1:2]
    
    def forward(self, x):
        x = self.conv1(x)
        x=self.activation(x)
        x = self.conv2(x)
        x =  self.agg(x) #bx1xWxH
        return x
    
    def training_step(self, batch, batch_idx):
        x1 = batch['image1']
        x2 = batch["image2"]

        with torch.no_grad():
            x1_encoded = self.sum_layers(self.feature_encoder.forward(x1))
            x2_encoded = self.sum_layers(self.feature_encoder.forward(x2))

        # x1_encoded.requires_grad = False
        # x2_encoded.requires_grad = False
        y1 = self.forward(x1_encoded)
        y2 = self.forward(x2_encoded)

        loss = self.loss(x1_encoded, x2_encoded, y1, y2, batch)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x1 = batch['image1']
        x2 = batch["image2"]

        with torch.no_grad():
            x1_encoded = self.sum_layers(self.feature_encoder.forward(x1))
            x2_encoded = self.sum_layers(self.feature_encoder.forward(x2))

        # x1_encoded.requires_grad = False
        # x2_encoded.requires_grad = False
        y1 = self.forward(x1_encoded)
        y2 = self.forward(x2_encoded)

        loss = self.loss(x1_encoded, x2_encoded, y1, y2, batch)

        self.log('validation_loss', loss)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def concat_layers(self, x_dict):
        return torch.cat([x_dict["early"], x_dict["middle"], x_dict["deep"]], 1) #bx48xWxH

    def sum_layers(self, x_dict):
        return x_dict["early"] + x_dict["middle"] + x_dict["deep"] #bx48xWxH

class MultiAttentionLayer(LightningModule):
    def __init__(self, feature_encoder):
        super().__init__()
        self.feature_encoder = feature_encoder
        if (REP_LOSS):
            self.loss = RepeatabilityLoss()
        else:
            self.loss = DistinctivenessLoss()

        self.early_attentions = nn.Conv2d(in_channels=self.feature_encoder.encoded_channels, \
                    out_channels=2, kernel_size=(1,1)) #bx2xWxH
        self.middle_attentions = nn.Conv2d(in_channels=self.feature_encoder.encoded_channels, \
                    out_channels=2, kernel_size=(1,1)) #bx2xWxH
        self.deep_attentions = nn.Conv2d(in_channels=self.feature_encoder.encoded_channels, \
                    out_channels=2, kernel_size=(1,1)) #bx2xWxH

    def softmax(self, ux):
        if ux.shape[1] == 1:
            x = F.softplus(ux)
            return x / (1 + x)  # for sure in [0,1], much less plateaus than softmax
        elif ux.shape[1] == 2:
            return F.softmax(ux, dim=1)[:,1:2]

    def forward(self, x):
        y = {}
        y["early"] = self.softmax(self.early_attentions(x["early"]))
        y["middle"] = self.softmax(self.middle_attentions(x["middle"]))
        y["deep"] = self.softmax(self.deep_attentions(x["deep"]))
        return y

    def training_step(self, batch, batch_idx):
        x1 = batch['image1']
        x2 = batch["image2"]

        with torch.no_grad():
            x1_encoded = self.feature_encoder.forward(x1)
            x2_encoded = self.feature_encoder.forward(x2)

        # x1_encoded.requires_grad = False
        # x2_encoded.requires_grad = False
        y1 = self.forward(x1_encoded)
        y2 = self.forward(x2_encoded)

        loss = torch.tensor(np.array([0], dtype=np.float32), device='cuda' if torch.cuda.is_available() else "cpu")

        for layer in x1_encoded.keys():
            loss = loss +  self.loss(x1_encoded[layer], x2_encoded[layer], y1[layer], y2[layer], batch)
        
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x1 = batch['image1']
        x2 = batch["image2"]

        with torch.no_grad():
            x1_encoded = self.feature_encoder.forward(x1)
            x2_encoded = self.feature_encoder.forward(x2)

        # x1_encoded.requires_grad = False
        # x2_encoded.requires_grad = False
        y1 = self.forward(x1_encoded)
        y2 = self.forward(x2_encoded)

        loss = torch.tensor(np.array([0], dtype=np.float32), device='cuda' if torch.cuda.is_available() else "cpu")

        for layer in x1_encoded.keys():
            loss = loss +  self.loss(x1_encoded[layer], x2_encoded[layer], y1[layer], y2[layer], batch)

        self.log('validation_loss', loss)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

if __name__ == "__main__":
    autoencoder = FeatureEncoder1.load_from_checkpoint("lightning_logs/version_2/checkpoints/epoch=56-step=8264.ckpt").requires_grad_(False)
    attentions = AttentionLayer2(autoencoder)
    if REP_LOSS:
        tb_logger = TensorBoardLogger('tb_logs', name='attention_model_repeatability_loss')
    else:
        tb_logger = TensorBoardLogger('tb_logs', name='attention_model_delf_distinctiveness_loss')
    trainer = pytorch_lightning.Trainer(logger=tb_logger, gpus=1 if torch.cuda.is_available() else None)
    dm = CorrespondenceDataModule()
    trainer.fit(attentions, dm)