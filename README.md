# Sauron

Repository of the paper **Sauron U-Net: Simple automated redundancy elimination in medical image segmentation via filter pruning**

### Table of Contents
* [1. Using Sauron](#1-using-sauron)
* [1.1 Prerequisites and data](#11-prerequisites-and-data)
* [1.2 Training](#12-training)
* [1.3 Computing the output](#13-computing-the-output)
* [1.4 Evaluation](#14-evaluation)
* [2. Experiments](#2-experiments)
* [2.1 Section 4.1: Benchmark](#21-section-41-benchmark)
* [2.2 Section 4.2: Clusterability](#22-section-42-clusterability)
* [2.3 Section 4.3: Feature maps interpretation](#23-section-43-feature-maps-interpretation)

### 1. Using Sauron

#### 1.1 Prerequisites and data
Libraries: We utilized Pytorch 1.7.1 and TorchIO 0.18.71.
Datasets: We utilized two publicly-available datasets: [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/) and [KiTS](https://kits19.grand-challenge.org/)
The exact train-val-test splits and data augmentation parameters are specified in the code (lib/data/*).

#### 1.2 Training
By specifying the dataset name, we can train a nnUNet model from scratch.

```
python train.py --data datasetName
```
The remaining parameters, such as the number of epochs, dataset splits and seeds, can be found in the function [parseArguments](lib/utils.py).

To continue training a model:
```
python train.py --data datasetName --epochs 500 --seed_train 42 --seed_data 42 --split 0.9:0.1 --epoch_start 400 --model_state path/model-400 --history path/to/previous/run
```
Particularly, --history expects the path that contains the pruned filters and other essential files that were saved during training.

#### 1.3 Computing the output
To generate the segmentations from a Sauron-pruned nnUNet model:

```
python predict.py --data datasetName --output output_path/predictions --model_state path/model-500 --original /path/to/dataset --in_filters path/in_filters --out_filters path/out_filters
```

--original expects the path to the original files of the dataset. This is important to guarantee that the segmentations will have the same voxel resolution.

#### 1.4 Evaluation
```
python evaluate.py --data datasetName --pred path/predictions --gt path/ground_truth --output output_path/results.json
```

### 2. Experiments

#### 2.1 Section 4.1: Benchmark
1. Sauron was run following the steps described above.
2. For Sauron ( ![formula](https://render.githubusercontent.com/render/math?math=\lambda=0) )
2.1 lib/loss: Set ![formula](https://render.githubusercontent.com/render/math?math=\lambda=0) in DS_CrossEntropyDiceLoss_Distance
2.2 train.py: leave callback._end_epoch_prune
2.3 lib/data/XXXDataset: dist_fun = "euc_norm"; imp_fun = "euc_rand"
3. For nnUNet
3.1 train.py: remove callback._end_epoch_prune
3.2 train.py: model = nnUNet(**cfg["architecture"])
3.3 lib/data/XXXDataset: dist_fun = ""; imp_fun = ""
3.4 lib/data/XXXDataset: use DS_CrossEntropyDiceLoss instead of DS_CrossEntropyDiceLoss_Distance

#### 2.2 Section 4.2: Clusterability
Store the feature maps by adding to train.py the callback _end_epoch_save_all_FMs and remove _end_epoch_prune to avoid pruning.

#### 2.3 Section 4.3: Feature maps interpretation
Load the trained models

```
...
model = Sauron(**params)
model.initialize(device="cuda", model_state=path, log=log, isSauron=True)
test_data = data.get("test")
with torch.no_grad():
  for sub_i, subject in enumerate(test_data):
    ...
    FMs = model.forward_saveFMs(image)
...

```
