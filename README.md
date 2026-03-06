# NeuroDiffTorch

This repository provides the official PyTorch implementation of the paper [NeuralCPA: A Deep Learning Perspective on Chosen-Plaintext Attacks](https://eprint.iacr.org/2026/328.pdf). It offers a unified training framework for neural differential distinguishers across multiple ciphers and model architectures, enabling them to perform the CPA game.

## Installation

The repository has been tested with **Python 3.8.20**.

### 1. Create a conda environment

```bash
conda create -n <env_name> python=3.8
conda activate <env_name>
```

### 2. Install dependencies

```bash
pip install pyyaml numpy scikit-learn psutil tqdm
```

Install PyTorch with CUDA following the [official instructions](https://pytorch.org/get-started/locally/), then verify GPU access:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Usage

### 1. Create a cipher dataset
Dataset configs are available for **SIMON**, **SPECK**, **LEA**, **HIGHT**, **XTEA**, **TEA**, **PRESENT**, **AES**, **KATAN**, and **CHACHA**.

```bash
python pipelines/create.py <dataset_config>
# e.g. python pipelines/create.py configs/datasets/speck_32_64.yaml
```

### 2. Train a neural differential distinguisher
The framework supports two neural distinguisher architectures **GohrNet** and **DBitNet**.
```bash
python pipelines/train.py <training_config>
# e.g. python pipelines/train.py configs/speck_32_64/n5_gohrnet.yaml
```

### 3. Play the CPA game
```bash
python attacks/cpa.py --exp <experiment_dir> --nr <cipher_rounds> --n <num_cpa_games> --ckpt <checkpoint_epoch>
# e.g. python attacks/cpa.py --exp output/speck_32_64/n5_gohrnet_4x16_20260126_051743 --nr 5 --n 100000 --ckpt 40
```

## Customization

New components can be added with custom implementations and config files.

### 1. Adding a New Cipher

To add a new cipher:

1. Implement the cipher in `/datasets/ciphers/`.
2. Import the cipher in `/datasets/ciphers/__init__.py`.
3. Create a dataset config in `/configs/datasets/`.

Example dataset config:

```yaml
data: speck_32_64_n5
data_path: 'data'

cipher: Speck_32_64           # cipher class name
n: 10**7                      # number of sample pairs
nr: 5                         # number of rounds
key_mode: 'random_fixed'      # encryption key mode
key: null                     # fixed key
diff: [0x0040, 0x0000]        # input difference
train_ratio: 0.9              # train/val split
negative_samples: 'real_encryption'  # negative sample mode
# batch_size: 10000           # optional dataset generation batch size
# seed: 42                    # optional random seed
```

- **key_mode**

  Controls how encryption keys are generated during dataset construction.

  - `random` Each sample pair uses a random key.
  - `random_fixed` All sample pairs share the same random key.
  - `input_fixed` The key provided in `key` is used for all samples.

- **negative_samples**

  Controls how negative samples are generated.

  - `real_encryption` Negative samples are produced from real encryptions of random plaintext pairs.
  - `random_bits` Negative samples are random bit sequences.

### 2. Adding a Neural Distinguisher

To add a new neural architecture:

1. Implement the model in `/distinguisher/models/`.
2. Import the model in `/distinguisher/models/__init__.py`.
3. Create a training config in `/configs/`.

Example training config:

```yaml
experiment:
  name: "n5_gohrnet_4x16"
  output_dir: "output/speck_32_64"
  # seed: 42

data:
  path: "data/speck_32_64_n5"      # dataset path

processor:
  swap_pairs: false                # swap ciphertext pair order
  xor_channel: false               # add XOR channel
  reshape: [4, 16]                 # input shape
  normalize: true                  # normalize to [-1,1]

model:
  type: "GohrNet"
  params:
    length: 16
    in_channels: 4
    n_filters: 32
    n_blocks: 10
    d1: 64
    d2: 64

training:
  epochs: 40                       # training epochs
  batch_size: 5000                 # batch size
  num_workers: 24                  # dataloader workers
  eval_interval: 5                 # eval interval
  checkpoint_interval: 20          # checkpoint interval
  device: "cuda:1"                 # training device
  use_amp: true                    # mixed precision
  # pretrained: "path/to/checkpoint.pt"   # optional pretrained checkpoint

  loss:
    type: "BCEWithLogitsLoss"
    params: {}
  
  optimizer:
    type: "AdamW"
    params:
      lr: 2e-3
      betas: [0.9, 0.999]
      weight_decay: 1e-5
  
  scheduler:
    type: "CosineAnnealingWarmRestarts"
    params:
      T_0: 18000
      T_mult: 1
      eta_min: 1e-4
```

- **pretrained**

  Loads model weights from an existing checkpoint to initialize training. This is commonly used for progressive round training. For example, when training an 8-round model for SPECK 32/64, a previously trained 7-round checkpoint can be used as initialization, allowing the model to build on the 7-round representation.

## Acknowledgements

We express our gratitude to the following open-source projects:

- [AutoND](https://github.com/Crypto-TII/AutoND): A Cipher-Agnostic Neural Training Pipeline with Automated Finding of Good Input Differences.
- [deep_speck](https://github.com/agohr/deep_speck): Improving Attacks on Round-Reduced Speck32/64 Using Deep Learning.

## Citation

If you use this repository in academic work, please cite:

```bibtex
@misc{cryptoeprint:2026/328,
      author = {Xuanya Zhu and Liqun Chen and Yangguang Tian and Gaofei Wu and Xiatian Zhu},
      title = {{NeuralCPA}: A Deep Learning Perspective on Chosen-Plaintext Attacks},
      howpublished = {Cryptology {ePrint} Archive, Paper 2026/328},
      year = {2026},
      url = {https://eprint.iacr.org/2026/328}
}
```
