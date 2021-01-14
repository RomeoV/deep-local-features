import torch
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger

from torch.nn import functional as F
import pytorch_lightning

from lib.loss import *
from lib.repeatability_loss import RepeatabilityLoss
from lib.autoencoder import *
from lib.train_shared_fe64 import *
REP_LOSS = False


class MultiAttentionLayerShared(LightningModule):
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
        # x1_encoded.requires_grad = False
        # x2_encoded.requires_grad = False

        x1 = self.feature_encoder.get_resnet_layers(batch["image1"])
        x2 = self.feature_encoder.get_resnet_layers(batch["image2"])

        z1 = {}
        x_hat1 = {}
        z2 = {}
        x_hat2 = {}
        for s in x1.keys():
            z1[s] = self.feature_encoder.encoder[s](x1[s])
            x_hat1[s] = self.feature_encoder.decoder[s](z1[s])
            z2[s] = self.feature_encoder.encoder[s](x2[s])
            x_hat2[s] = self.feature_encoder.decoder[s](z2[s])

        loss = sum((F.mse_loss(x1[s], x_hat1[s]) + F.mse_loss(x2[s], x_hat2[s]) + self.feature_encoder.cfactor * self.feature_encoder.correspondence_loss(z1[s],z2[s], batch)) for s in x1.keys())
    
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
        z2 = {}
        x_hat2 = {}
        for s in x1.keys():
            z1[s] = self.feature_encoder.encoder[s](x1[s])
            x_hat1[s] = self.feature_encoder.decoder[s](z1[s])
            z2[s] = self.feature_encoder.encoder[s](x2[s])
            x_hat2[s] = self.feature_encoder.decoder[s](z2[s])

        loss = sum((F.mse_loss(x1[s], x_hat1[s]) + F.mse_loss(x2[s], x_hat2[s]) + self.feature_encoder.cfactor * self.feature_encoder.correspondence_loss(z1[s],z2[s], batch)) for s in x1.keys())
    
        y1 = self.forward(z1)
        y2 = self.forward(z2)

        for layer in x1.keys():
            loss = loss +  self.loss(z1[layer], z2[layer], y1[layer], y2[layer], batch)
        
        self.log('validation_loss', loss)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
if __name__=="__main__":
    autoencoder = CorrespondenceEncoder()
    attentions = MultiAttentionLayer2Shared(autoencoder)
    if REP_LOSS:
        tb_logger = TensorBoardLogger('tb_logs', name='attention_model_repeatability_loss')
    else:
        tb_logger = TensorBoardLogger('tb_logs', name='cfe64_multi_attention_model_distinctiveness+_lossN32_lambda03_sm_lowmargin')
    trainer = pytorch_lightning.Trainer(logger=tb_logger, gpus=1 if torch.cuda.is_available() else None)
    dm = CorrespondenceDataModule()
    trainer.fit(attentions, dm)