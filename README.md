A Multi-Strategy Fusion-Based Approach for Pipeline Defect Segmentation and Quantification

Official PyTorch implementation of the paper:

“A Multi-Strategy Fusion-Based Approach for Pipeline Defect Segmentation and Quantification”

📌 1. Overview

This repository provides the full implementation used in the paper, including:

Network architecture

Training and testing scripts

Dataset preprocessing

Quantitative evaluation metrics

Generated defect point cloud analysis

This implementation allows full reproduction of the experimental results reported in the manuscript.

📄 2. Reproducibility Statement

All experimental results in the paper can be reproduced using:

The source code in this repository

The dataset archived at:
https://doi.org/10.5281/zenodo.18859975

Random seed used in experiments:

seed = 2024

Training/test split:

80% training

20% testing

Number of sampled points:

2048 per point cloud

All reported metrics are averaged over 5 independent runs.

🔧 3. Environment Requirements

Python 3.8

PyTorch ≥ 1.4

NumPy

Visdom

CUDA (recommended)

Install dependencies:

pip install torch torchvision numpy visdom

📂 4. Dataset Structure

After downloading the dataset from Zenodo, organize as:

dataset/
│
├── Defect_Type_A/
│   ├── sample_001.npy
│   ├── sample_002.npy
│
├── Defect_Type_B/

Each file contains:

[X, Y, Z]

During loading:

Random sampling to 2048 points

Normalization to unit sphere

⚙️ 5. Training
Step 1: Start Visdom

python -m visdom.server -port 8097

Step 2: Run Training

python train.py \
--dataset_path /path/to/dataset \
--class_choice all \
--batch_size 32 \
--epochs 2000 \
--gpu 0

🧪 6. Testing / Evaluation

To reproduce quantitative results:

python test.py \
--dataset_path /path/to/dataset \
--model_path model/ckpt/generator_2000.pt

Evaluation metrics implemented:

Chamfer Distance (CD)

Earth Mover’s Distance (EMD)

Structural similarity metrics

Results will be saved to:

results/

📊 7. Output Files

Model checkpoints:

model/ckpt/

Generated samples:

model/generated/

Quantitative evaluation:

results/metrics.csv

📎 8. Citation

If you use this code or dataset, please cite:

@article{yourpaper2026,
  title={A Multi-Strategy Fusion-Based Approach for Pipeline Defect Segmentation and Quantification},
  journal={Discover Applied Sciences},
  year={2026}
}

Dataset DOI:
10.5281/zenodo.18859975
