# NeuroDiffTorch
A PyTorch framework for neural differential distinguishers in block cipher cryptanalysis.

## Usage

### 1. Create cipher dataset
```bash
python pipelines/create.py configs/datasets/speck_32_64.yaml
```

### 2. Train neural distinguisher
```bash
python pipelines/train.py configs/speck_32_64/speck_32_64_n5_gohrnet.yaml
```

## Acknowledgements

Gratitude to the following open-source projects:
- [AutoND](https://github.com/Crypto-TII/AutoND): A Cipher-Agnostic Neural Training Pipeline with Automated Finding of Good Input Differences.
- [deep_speck](https://github.com/agohr/deep_speck): Improving Attacks on Round-Reduced Speck32/64 Using Deep Learning.
