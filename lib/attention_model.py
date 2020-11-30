import torch
from torch import nn
from pytorch_lightning.core.lightning import LightningModule


from torch.nn import functional as F
import pytorch_lightning

from lib.loss import SimilarityLoss
from lib.autoencoder import FeatureEncoder

class AttentionLayer(LightningModule):
    def __init__(self, feature_encoder):
        super().__init__()
        self.feature_encoder = feature_encoder
        self.loss = SimilarityLoss()

    def softmax(self, ux):
        if ux.shape[1] == 1:
            x = F.softplus(ux)
            return x / (1 + x)  # for sure in [0,1], much less plateaus than softmax
        elif ux.shape[1] == 2:
            return F.softmax(ux, dim=1)[:,1:2]
    
    def forward(self, x):
        x = nn.Conv2d(in_channels=self.feature_encoder.encoded_channels, \
            out_channels=2, kernel_size=(1,1)); #bx2xWxH
        x = softmax(x) #bx1xWxH
        return x
    
    def training_step(self, batch, batch_idx):
        x1 = batch['image1']
        x2 = batch["image2"]

        x1_encoded = self.feature_encoder.forward(x1)
        x2_encoded = self.feature_encoder.forward(x2)

        # We need to concatenate the features of {x1, x2}_encoded here

        x1_encoded.requires_grad = False
        x2_encoded.requires_grad = False
        y1 = self.forward(x1_encoded)
        y2 = self.forward(x2_encoded)

        loss = self.loss(y1,y2, correspondence)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):

        x1 = batch['image1']
        x2 = batch["image2"]

        x1_encoded = self.auto_encoder.forward(x1)
        x2_encoded = self.auto_encoder.forward(x2)

        # We need to concatenate the features of {x1, x2}_encoded here
        
        x1_encoded.requires_grad = False
        x2_encoded.requires_grad = False

        y1 = self.forward(x1_encoded)
        y2 = self.forward(x2_encoded)

        loss = self.loss(y1,y2, correspondence)
        
        self.log('validation_loss', loss)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
