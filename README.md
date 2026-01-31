# wcpc-gan
Pipeline Defect Segmentation and Quantification

This repository provides the official implementation of the paper:

“A Multi-Strategy Fusion-Based Approach for Pipeline Defect Segmentation and Quantification”

📌 Overview

This project focuses on automated defect segmentation and quantitative assessment for underground drainage pipelines. The proposed framework integrates image enhancement, defect segmentation, depth estimation, and defect quantification to achieve an end-to-end inspection pipeline.

🧠 Method Highlights

Multi-strategy feature fusion for defect segmentation

Robust post-processing to reduce color mixing and boundary ambiguity

Quantitative defect assessment based on geometric reconstruction

📂 Repository Structure
├── data/
│   ├── sample_dataset/        # Representative subset of the dataset
│   └── README.md              # Data description
├── models/                    # Network architectures
├── scripts/                   # Training and evaluation scripts
├── utils/                     # Utility functions
├── configs/                   # Configuration files
└── README.md

📊 Dataset Availability

Due to data volume and usage constraints, the complete dataset used in the paper cannot be fully released at this stage.
To ensure reproducibility, we provide:

A representative subset of the dataset

Label annotations and class definitions

Data preprocessing scripts

This subset is sufficient to reproduce the main experimental pipeline and validate the effectiveness of the proposed method.

The dataset structure follows:

img.png: original pipeline images

label.png: pixel-wise segmentation labels

label_name.txt: class definitions

Additional data will be released in future updates when possible.

🚀 Getting Started

Clone the repository:

git clone https://github.com/yourname/yourrepo.git


Install dependencies:

pip install -r requirements.txt


Run training:

python train.py --config configs/default.yaml

🔁 Reproducibility

All experiments reported in the paper can be reproduced using the provided code, configuration files, and sample dataset.

📬 Contact

If you have any questions regarding the code or data, please feel free to contact:

Yao Maolin
Email: (your email)
