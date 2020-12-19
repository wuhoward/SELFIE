# PyTorch implementation of Selfie: Self-supervised Pretraining for Image Embedding

This repository implements the paper [Selfie](https://arxiv.org/abs/1906.02940). We reuse the Preact-ResNet model from this [repository](https://github.com/bearpaw/pytorch-classification/blob/master/models/cifar/preresnet.py).

### Run Selfie Pretraining

```bash
python train_selfie.py --lr 0.4
# or run with noisy patches instead of skipping them
python train_selfie.py --lr 0.4 --all-patch
```

### Finetuning the Selfie pretrained model with an additional ResNet block

```bash
python train_full.py --lr 0.01 --resume-selfie best.pth
```

### Training the Baseline model from scratch

```bash
python train_full.py --lr 0.2
```

