# ProtLoc

## Installation

Recommended: create a new environment from pypi and install the package.

```bash
pip install -r requirements.txt
```

## Prepare Data

We use the [tensorflow-datasets](https://www.tensorflow.org/datasets) as default. You just run the code, the dataset will be downloaded and prepared automatically. As the dataset is large, it may take a while (maybe 1 day) to download.

## Pre-training

```bash
python train.py --config_file configs/pretrain.yaml --workdir workdir/pretrain
```

## Fine-tuning

Please update the pretrained model path in the config file before running the following command.

```bash
python train.py --config_file configs/finetune.yaml --workdir workdir/finetune
```

## Evaluation

```bash
python test.py --workdir workdir/finetune
```
