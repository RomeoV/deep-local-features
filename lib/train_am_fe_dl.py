import torch
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger

from torch.nn import functional as F
import pytorch_lightning

from lib.loss import *
from lib.repeatability_loss import RepeatabilityLoss
from lib.autoencoder import *
from lib.attention_model import *


if __name__ == "__main__":
    autoencoder = FeatureEncoder.load_from_checkpoint("lightning_logs/version_4/checkpoints/epoch=70-step=10294.ckpt").requires_grad_(False)
    attentions = MultiAttentionLayer(autoencoder)
    if REP_LOSS:
        tb_logger = TensorBoardLogger('tb_logs', name='attention_model_repeatability_loss')
    else:
        tb_logger = TensorBoardLogger('tb_logs', name='fe_multi_attention_model_distinctiveness_loss')
    trainer = pytorch_lightning.Trainer(logger=tb_logger, gpus=1 if torch.cuda.is_available() else None)
    dm = CorrespondenceDataModule()
    trainer.fit(attentions, dm)