# Using Multi-Level Convolutional Information for Scale- and Viewpoint-Robust Local Features

## Abstract
> Fast and robust key point detection and feature matching is an important task for many problems in computer vision, for example Image Retrieval and Augmented Reality.
Two key issues arise: Key point detectors in real-time applications have to be both fast to compute and robust to change in illumination, viewpoint and scale.
In this work we aim to increase the robustness of feature descriptors against scale and viewpoint changes while maintaining similar performance otherwise.
To this end, we expand on previous work, using a pretrained ResNet backbone and add an attention layer for keypoint selection, which we train directly on the quality of the keypoints.
Critical to our goal, we forward multi-level convolutional activations directly to the final attention layer, bypassing further transformations and thus combining local with global information in the descriptor generation.

  Architecture             | Correspondence examples
:-------------------------:|:-------------------------:
![Architecture](img/ArchitectureFinal.png) | ![Correspondence examples](img/CorrespondenceExamples.png)

## Directory layout
Here, an overview over the files is given. The main components are the **dataset**, **architecture**, **losses** and **benchmarking**.
```
.
├── checkpoints
├── externals
├── hpatches_sequences                <--- Benchmarking files
├── lib
│   ├── feature_extractor
│   │   └── feature_extraction.py
│   ├── attention_model.py            <--- Attention and training
│   ├── autoencoder.py                <--- Autoencoder and training
│   ├── correspondence_datamodule.py  <--- Correspondence handler
│   ├── extraction_model.py           <--- Feature extraction
│   ├── loss.py                       <--- Different loss implementations
│   └── megadepth_dataset.py          <--- Dataset handler
├── notebooks
├── scripts
│   ├── extract_features.py
│   └── plot_keypoints.py
├── load_script_leonhard.sh
├── README.md
├── requirements.txt
└── run_bench.sh
```

### Setup
Create a new conda environment and install the packages from *requirements.txt*
```bash
conda create -n DL
conda activate DL
conda install --file requirements.txt
```

### Load Model & Evaluate with HPatches
First go to *hpatches_sequences* and run
```
sh download.sh
sh download_cache.sh
```
to download the HPatches dataset.

To extract keypoints on the dataset run from the project root folder
```
python extract_features.py --nouse_nms --attention_ckpt cfe64_multi_attention_model2_d2netloss
```
The results will be saved with the extension **.our-model**. In order to use another extension use `--output_extension .some-extension` or use `--smart_name` in order to generate an extension based on the used model and parameters.

Other useful parameters: 
* `--nogpu` if you don't have a GPU 
* `--encoder_ckpt correspondence_encoder_lr1e3` to chose another encoder checkpoint
By default, checkpoints are automatically downloaded from Polybox, this, however, requires authentication. As alternative, you can load from a local file as follows: 
* `--load_from_folder` The checkpoint must be in *checkpoints/checkpoint_name.ckpt* as set by --attention_ckpt

##### Evaluate
Run *HPatches-Benchmark.ipynb* in *hpatches_sequence*. Add the used extensions to methods.

### Available Checkpoints

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

## Similar Work
[DELF & DELG](https://github.com/tensorflow/models/tree/master/research/delf)  
[DELF PyTorch :) ](https://github.com/nashory/DeLF-pytorch)  
[D2-Net](https://github.com/mihaidusmanu/d2-net)  
[Deep Image Retrieval](https://github.com/naver/deep-image-retrieval)  
