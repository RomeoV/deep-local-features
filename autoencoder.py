import torch
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from AE_data_wrangling import AutoencoderDataModule

from torch.nn import functional as F
import pytorch_lightning

class LitAutoEncoder(LightningModule):
  def __init__(self):
    super().__init__()
    self.encoder = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=10, kernel_size=(64,64)),
    )
    self.decoder = nn.Sequential(
        nn.ConvTranspose2d(in_channels=10, out_channels=64, kernel_size=(64,64)),
    )
  
  def forward(self, x):
    embedding = self.encoder(x)
    return nn.ReLU(embedding)
  
  def training_step(self, batch, batch_idx):
    x = batch['layer1_conv1']
    z = self.encoder(x)
    x_hat = self.decoder(z)
    loss = F.mse_loss(x, x_hat)
    self.log('train_loss', loss)
    return loss

  def validation_step(self, batch, batch_idx):
    x = batch['layer1_conv1']
    z = self.encoder(x)
    x_hat = self.decoder(z)
    val_loss = F.mse_loss(x, x_hat)
    self.log('val_loss', val_loss)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    return optimizer

if __name__ == "__main__":
    autoencoder = LitAutoEncoder()
    trainer = pytorch_lightning.Trainer(gpus=1 if torch.cuda.is_available() else None)
    dm = AutoencoderDataModule()
    trainer.fit(autoencoder, dm)
