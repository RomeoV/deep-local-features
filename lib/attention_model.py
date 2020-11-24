import torch
from torch import nn
from pytorch_lightning.core.lightning import LightningModule


from torch.nn import functional as F
import pytorch_lightning

from lib.loss import SimilarityLoss
from lib.autoencoder import LitAutoEncoder

class AttentionLayer(LightningModule):
    def __init__(self, auto_encoder):
        super().__init__()
        self.auto_encoder = auto_encoder
        self.loss = SimilarityLoss()

    def softmax(self, ux):
        if ux.shape[1] == 1:
            x = F.softplus(ux)
            return x / (1 + x)  # for sure in [0,1], much less plateaus than softmax
        elif ux.shape[1] == 2:
            return F.softmax(ux, dim=1)[:,1:2]
    
    def forward(self, x):
        x = nn.Conv2d(in_channels=10, out_channels=2, kernel_size=(1,1)); #bx2xWxH
        x = softmax(x) #bx1xWxH
        return x
    
    def training_step(self, batch, batch_idx):
        correspondence = batch['correspondence']

        x1 = batch["activations1"]['layer1_conv1']
        x2 = batch["activations2"]['layer1_conv1']

        x1_encoded = self.auto_encoder.forward(x1)
        x2_encoded = self.auto_encoder.forward(x2)
        y1 = self.forward(x1_encoded)
        y2 = self.forward(x2_encoded)

        loss = self.loss(y1,y2, correspondence)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        correspondence = batch['correspondence']

        x1 = batch["activations1"]['layer1_conv1']
        x2 = batch["activations2"]['layer1_conv1']

        x1_encoded = self.auto_encoder.forward(x1)
        x2_encoded = self.auto_encoder.forward(x2)
        y1 = self.forward(x1_encoded)
        y2 = self.forward(x2_encoded)

        loss = self.loss(y1,y2, correspondence)
        
        self.log('validation_loss', loss)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

if __name__ == "__main__":
    auto_encoder = LitAutoEncoder()
    attention = AttentionLayer(auto_encoder)
    trainer = pytorch_lightning.Trainer(gpus=1 if torch.cuda.is_available() else None)
    dm = AutoencoderDataModule()
    trainer.fit(autoencoder, dm)
