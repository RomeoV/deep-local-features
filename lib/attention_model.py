import torch
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger

from torch.nn import functional as F
import pytorch_lightning

from lib.loss import *
#from lib.repeatability_loss import RepeatabilityLoss
from lib.autoencoder import *
from lib.train_shared_fe64 import *

class MultiAttentionLayer2(LightningModule):
    def __init__(self, feature_encoder):
        super().__init__()
        self.feature_encoder = feature_encoder
        self.loss = DistinctivenessLoss()

        self.early_attentions = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_encoder.encoded_channels,
                      out_channels=512, kernel_size=(1, 1)),  # bx2xWxH
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, \
                      out_channels=1, kernel_size=(1, 1)),  # bx2xWxH
            nn.Softplus(beta=1, threshold=20),
        )
        self.middle_attentions = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_encoder.encoded_channels,
                      out_channels=512, kernel_size=(1, 1)),  # bx2xWxH
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, \
                      out_channels=1, kernel_size=(1, 1)),  # bx2xWxH
            nn.Softplus(beta=1, threshold=20),
        )
        self.deep_attentions = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_encoder.encoded_channels,
                      out_channels=512, kernel_size=(1, 1)),  # bx2xWxH
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, \
                      out_channels=1, kernel_size=(1, 1)),  # bx2xWxH
            nn.Softplus(beta=1, threshold=20),
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
        y["early"] = self.early_attentions(x["early"])
        y["middle"] = self.middle_attentions(x["middle"])
        y["deep"] = self.deep_attentions(x["deep"])
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

        loss = torch.tensor(np.array(
            [0], dtype=np.float32), device='cuda' if torch.cuda.is_available() else "cpu")

        for layer in x1_encoded.keys():
            loss = loss + \
                self.loss(x1_encoded[layer], x2_encoded[layer],
                          y1[layer], y2[layer], batch)

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

        loss = torch.tensor(np.array(
            [0], dtype=np.float32), device='cuda' if torch.cuda.is_available() else "cpu")

        for layer in x1_encoded.keys():
            loss = loss + \
                self.loss(x1_encoded[layer], x2_encoded[layer],
                          y1[layer], y2[layer], batch)

        self.log('validation_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


if __name__ == "__main__":
    REP_LOSS=False
    autoencoder = CorrespondenceEncoder.load_from_checkpoint("tb_logs/correspondence_encoder_lr1e3/version_0/checkpoints/epoch=7-step=1159_interm.ckpt").requires_grad_(False)
    attentions = MultiAttentionLayer2(autoencoder)

    tb_logger = TensorBoardLogger('tb_logs', name='cfe64_multi_attention_model2_distinctiveness+_loss')
    trainer = pytorch_lightning.Trainer(logger=tb_logger, gpus=1 if torch.cuda.is_available() else None)
    dm = CorrespondenceDataModule()
    trainer.fit(attentions, dm)
