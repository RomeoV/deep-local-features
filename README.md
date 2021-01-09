#Improving Scale- and Viewpoint Invariance of Learned LocalFeatures

### Setup
Create a new conda environment and install the packages from *requirements.txt*
```bash
conda create -n DL
conda activate DL
conda install --file requirements.txt
```

### Run your .py files in lib
go to the root directory of the repo
```
python -m lib.autoencoder
```

### Run tensorboard on leonhard and open locally
First, make sure you have the right version of python loaded (with tensorflow etc).
My script is this one:
```
# Filename: load_script.sh
# Activate using `source ./load_script.sh`
module load gcc/6.3.0
module load python_gpu/3.8.5
# pip install --user tk-tools
module load tk/8.6.6
module load hdf5/1.10.1
```
Then, you can open tensorboard and pass a free port. You can also try to leave out the port and copy the one it gives you.
```
tensorboard --logdir ./tb_logs --port 6660
```
Finally, on your local computer, you can forward the port from the cluster to your local computer:
```
ssh -NL 6001:localhost:6660 eth-id@login.leonhard.ethz.ch
```
Now you can open `http://localhost:6001` in your local browser and check out your tensorflow.

### TODOs
Architectures:

FeatureEncoder1: Trained (conv1 layers)
FeatureEncoder3: Trained (conv3 layers)
FeatureEncoder: Non-finished trainining (more channels)

AttentionLayers:
AttentionLayer (R2D2 architecture, 34 parameter)
    Sum featuremaps (concat_layers())
    concat featuremaps (sum_layers())

AttentionLayer2 (DeLF architecture)
MultiAttentionLayer

Loss:
D2Net loss (loss.py)
distinctiveness loss (loss.py)
repeatability loss (repeatability_loss.py)


Trained full models:
Works:
FeatureEncoder1 + AttentionLayerSum + distinctiveness-loss (path to weightfile)
FeatureEncoder1 + AttentionLayer2Sum + distinctiveness-loss (path to weightfile)
FeatureEncoder1 + MultiAttentionLayer + distinctiveness-loss (path to weightfile)

Works not:
FeatureEncoder1 + AttentionLayerConcat + D2Net

To Test:
FeatureEncoder1 + MultiAttentionLayer + repeatability-loss

FeatureEncoder + AttentionLayerSum + distinctiveness-loss
FeatureEncoder + MultiAttentionLayer + distinctiveness-loss 

FeatureEncoder3 + AttentionLayerSum + distinctiveness-loss (Romeo)
FeatureEncoder3 + MultiAttentionLayer + distinctiveness-loss (Romeo)

ToDo:
L2 Normalization
K in Distinctiveness Loss

Report:
Visualization Autoencoder: (Romeo)
    Early: Resnet -> Encoded features -> Decoded features
    Middle: Resnet -> Encoded features -> Decoded features
    Deep: Resnet -> Encoded features -> Decoded features

Visualization AttentionLayer: (evtl. mehrere im MultiAttentionLayer) 
    Image + Attention Layer on top (similar to R2D2)

Keypoints visualization: (Mark)
    1 Bild + keypoints (input: [x1, y1], [x2, y2], ...)
    Matching: 2 Bilder
### Goals

### Similar Work
[DELF & DELG](https://github.com/tensorflow/models/tree/master/research/delf)  
[DELF PyTorch :) ](https://github.com/nashory/DeLF-pytorch)  
[D2-Net](https://github.com/mihaidusmanu/d2-net)  
[Deep Image Retrieval](https://github.com/naver/deep-image-retrieval)  
