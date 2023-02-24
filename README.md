# Fair Attribute Completion on Graph with Missing Attributes

A PyTorch implementation of "Fair Attribute Completion on Graph with Missing Attributes"

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
