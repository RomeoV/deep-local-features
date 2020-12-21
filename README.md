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

### Goals

### Similar Work
[DELF & DELG](https://github.com/tensorflow/models/tree/master/research/delf)  
[DELF PyTorch :) ](https://github.com/nashory/DeLF-pytorch)  
[D2-Net](https://github.com/mihaidusmanu/d2-net)  
[Deep Image Retrieval](https://github.com/naver/deep-image-retrieval)  
