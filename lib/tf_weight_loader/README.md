## Tf Weight Loading 
In order for this script to work, we need to make the pre-trained weight a checkpoint. Luckily, this only requires one additional file. Download and extract the weights from [the delg repo](https://github.com/tensorflow/models/tree/master/research/delf). Then copy the 'checkpoint' file from this directory into the 'variables' folder of the data you just extracted. 

#### Running the tests 
From the repo root run 
```
python3 lib/tf_weight_loader/tests.py --checkpoint_path /path/to/checkpoint
```
