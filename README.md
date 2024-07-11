# Fair Attribute Completion on Graph with Missing Attributes

A PyTorch implementation of "Fair Attribute Completion on Graph with Missing Attributes"
[\[Paper\]](https://openreview.net/pdf?id=9vcXCMp9VEp)

## Requirements

```
torch==1.12.0
DGL=0.9.0
scikit-learn==1.1.1
```
or you can directly create a conda environment by environment.yml
```
conda env create -f environment.yml
```

## Model Training
1. We only care about the epochs that the accuracy and roc score are higher than the thresholds (defined by --acc and --roc).
2. We will select the epoch whose summation of parity and equal opportunity is the smallest.

To reproduce the performance reported in the paper, you can run the bash files in folder `src\scripts`.
```
bash scripts/pokec_z/train_fairAC.sh
```

## Acknowledgment
We thank @oxkitsune for the detailed [reimplementation](https://github.com/oxkitsune/fact).

## Citation
If you find this repo useful, please consider citing:
```
@inproceedings{
guo2023fair,
title={Fair Attribute Completion on Graph with Missing Attributes},
author={Dongliang Guo and Zhixuan Chu and Sheng Li},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=9vcXCMp9VEp}
}
```
