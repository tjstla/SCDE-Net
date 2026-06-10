# SCDE-Net

This repository contains the PyTorch implementation of **SCDE-Net: Strip-Context and Dilation-Enhanced Deep Unfolding Network for Infrared Small Target Detection**.

SCDE-Net builds on a deep unfolding RPCA framework and introduces:

- Strip Context Background Module (SCBM) for directional background context modeling.
- Dilated Context Sparse Module (DCSM) for enlarged local contrast perception.
- Adaptive Residual Fusion Module (ARFM) for component-aware reconstruction across unfolding stages.

## Environment

```bash
conda create -n scdenet python=3.9
conda activate scdenet
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Dataset Preparation

Please download the public datasets from their official sources and place them under `datasets/`.

Expected structure:

```text
datasets/
  IRSTD-1k/
    train/
      images/
      masks/
    test/
      images/
      masks/
  NUDT-SIRST/
    train/
      images/
      masks/
    test/
      images/
      masks/
  SIRSTv1/
    PNGImages/
    Splits/
    SIRST/
      BinaryMask/
```

Large datasets, checkpoints, generated results, logs, and visualization outputs are intentionally not included in this repository.

## Training

Train on IRSTD-1K:

```bash
python train.py --net-name scdenet --dataset irstd1k --batch-size 8 --epoch 400 --lr 1e-4 --base-dir train_logs
```

Train on NUDT-SIRST:

```bash
python train.py --net-name scdenet --dataset nudt --batch-size 8 --epoch 400 --lr 1e-4 --base-dir train_logs
```

Train on SIRST V1:

```bash
python train.py --net-name scdenet --dataset SIRSTv1 --batch-size 8 --epoch 400 --lr 1e-4 --base-dir train_logs
```

## Inference

Put a checkpoint under `checkpoints/`, then run inference with the corresponding dataset name:

```bash
python inference.py \
  --dataset IRSTD-1k \
  --checkpoint checkpoints/SCDE_IRSTD1K.pkl \
  --device cuda:0
```

Predicted masks and `.mat` files are saved under `result/<dataset>/`.


## Citation

If this code is useful for your research, please cite the corresponding paper after publication.

## License

No license file has been added yet. Please add an appropriate open-source license before making this repository public.

## Acknowledgements

This implementation is released as the official PyTorch code for SCDE-Net.
