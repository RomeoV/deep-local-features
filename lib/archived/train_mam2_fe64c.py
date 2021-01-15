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
from lib.train_shared_fe64 import *

REP_LOSS=False
if __name__ == "__main__":
    autoencoder = CorrespondenceEncoder.load_from_checkpoint("tb_logs/correspondence_encoder_lr1e3/version_0/checkpoints/epoch=7-step=1159_interm.ckpt").requires_grad_(False)
    attentions = MultiAttentionLayer2(autoencoder)
    if REP_LOSS:
        tb_logger = TensorBoardLogger('tb_logs', name='attention_model_repeatability_loss')
    else:
        tb_logger = TensorBoardLogger('tb_logs', name='cfe64_multi_attention_model2_distinctiveness+_loss')
    trainer = pytorch_lightning.Trainer(logger=tb_logger, gpus=1 if torch.cuda.is_available() else None)
    dm = CorrespondenceDataModule()
    trainer.fit(attentions, dm)
