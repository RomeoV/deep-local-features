#Improving Scale- and Viewpoint Invariance of Learned LocalFeatures

### Setup
Create a new conda environment and install the packages from *environment.yml*:
```bash
conda env create -n DL -f environment.yml
conda activate DL
```
If you are using Leonhard for training, you can run
```
source load_script_leonhard.sh
```
Note that if you want wo train you need to download the MegaDepth dataset and set 
```
export MegaDepthDatasetPath="/path/to/MegaDepthDataset"
```

### Load Model & Evaluate with HPatches

Already evaluated models can be seen in *hpatches_sequences/HPatches-Benchmark-Comparison.ipynb* and in *HPatches-Benchmark-Losses.ipynb*. The results for them are cached in *hpatches_sequences/cache*, so they notebook can be rerun yielding the same results.

##### Download HPatches
First go to *hpatches_sequences* and run
```bash
sh download.sh
```
to download the HPatches dataset.

##### Extract Keypoints

To extract keypoints on the dataset run from the project root folder
```
python extract_features.py --cfe64_multi_attention_model2_distinctiveness+_lossN2_lambda05_sm_SHARED --load_from_folder
```
The results will be saved with the extension **.our-model**. In order to use another extension use `--output_extension .some-extension` or use `--smart_name` in order to generate an extension based on the used model and parameters. The generated extensions are written to *checkpoints/extensions.txt*.

Other useful parameters: 
* `--nogpu` if you don't have a GPU 
<!--* `--encoder_ckpt correspondence_encoder_lr1e3` to chose another encoder checkpoint
 As alternative, you can load from a local file as follows: -->
* `--load_from_folder` The checkpoint must be in *checkpoints/checkpoint_name.ckpt* as set by --attention_ckpt. By default, checkpoints are automatically downloaded from Polybox. This, however, requires authentication. If you want to use the automatic download, please contact philipp.lindenberger@math.ethz.ch. Alternatively, you can download the checkpoints from https://polybox.ethz.ch/index.php/s/LlnekeNyCa0QJ5I manually, add them with the correct name to the *checkpoints/* folder and use the `--load_from_folder` parameter.

##### Evaluate
Run *HPatches-Benchmark.ipynb* in *hpatches_sequence*. Add the used extensions the *methods* and *names* field in the notebook. Set *visualize * and *use_ransac* (described in the papaer as refined) to *True* or *False*. To make sure that the evaluation is executed, make sure that the extension you want to evaluate is not contained in the cache folder (just delete it). 

##### Available Attention Checkpoints
Attention checkpoints also contain weights for the encoder. The correct attention layer for each checkpoint is automatically loaded by *extract_features.py* when `python extract_features.py --attention_ckpt CHECKPOINT`. Available checkpoints are:
* cfe64_multi_attention_model2_d2netloss
* cfe64_multi_attention_model2_d2netloss_backprop
* cfe64_multi_attention_model_d2netloss
* cfe64_multi_attention_model_d2netloss_backprop
* cfe64_multi_attention_model_distinctiveness+_loss
* cfe64_multi_attention_model_distinctiveness+_lossN8_l1
* cfe64_multi_attention_model_distinctiveness+_lossN32_l1
* multi_attention_model_distinctiveness_loss
* cfe64_multi_attention_model2_distinctiveness+_lossN16_lambda01_sm_lowmargin_SHARED
* cfe64_multi_attention_model2_distinctiveness+_lossN16_lambda01_lowmargin
* cfe64_multi_attention_model2_distinctiveness+_lossN8_lambda01_sm_SHARED
* cfe64_multi_attention_model2_distinctiveness+_lossN8_lambda01_sm
* cfe64_multi_attention_model2_distinctiveness+_lossN8_lambda1
* cfe64_multi_attention_model2_distinctiveness+_lossN2_lambda05_sm_SHARED
* cfe64_multi_attention_model2_distinctiveness+_lossN2_lambda05_sm
* cfe64_multi_attention_model2_distinctiveness+_lossN64_lambda1
* cfe64_multi_attention_model2_distinctiveness+_lossN32_lambda1
* cfe64_multi_attention_model2_distinctiveness+_lossN16_lambda01_sm
* cfe64_multi_attention_model2_distinctiveness+_loss

### Training
TODO

TODO

TODO

### Run your .py files in lib
go to the root directory of the repo
```bash
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
