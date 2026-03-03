A Multi-Strategy Fusion-Based Approach for Pipeline Defect Segmentation and Quantification

This repository contains the official PyTorch implementation of the paper:

“A Multi-Strategy Fusion-Based Approach for Pipeline Defect Segmentation and Quantification”

The project provides a lightweight, pure PyTorch framework for 3D pipeline defect learning and quantitative analysis.

📌 Overview

This framework is designed for:

3D pipeline defect point cloud learning

Structural defect generation

Quantitative defect analysis

Easy integration with custom datasets

The implementation is minimal, flexible, and easy to extend.

🔧 1. Environment Preparation
Requirements

Python 3.x

PyTorch ≥ 1.4 (CUDA recommended)

NumPy

Visdom

Install Dependencies
pip install torch torchvision numpy visdom

A CUDA-enabled GPU is highly recommended for training.

📂 2. Custom Dataset Preparation

The project uses a custom data loader (data_benchmark.py) that directly supports .npy and .txt files.

Required Directory Structure
/your_custom_dataset_path/
│
├── Defect_Type_A/
│   ├── defect_001.npy
│   ├── defect_002.txt
│
├── Defect_Type_B/
│   ├── defect_001.npy
Data Requirements

Each file must contain 3D coordinates: (X, Y, Z)

During training, the loader will:

Randomly sample to 2048 points

Normalize point clouds automatically

⚙️ 3. Configuration Settings

All hyperparameters are defined in:

arguments.py

You can either modify the file directly or pass parameters via the command line.

Important Arguments
Argument	Description	Default
--dataset_path	Absolute dataset root path (Required)	None
--class_choice	Specific defect folder or all	all
--point_num	Number of sampled points	2048
--batch_size	Batch size	32
--epochs	Training epochs	2000
--gpu	GPU index	0
🚀 4. Training the Model
Step 1: Start Visdom Server

Open a new terminal:

python -m visdom.server -port 8097

Then open your browser:

http://localhost:8097

You will see:

Generator loss

Discriminator loss

3D point cloud visualization

Step 2: Launch Training

Run the following command:

python train.py \
--dataset_path /absolute/path/to/your/dataset \
--class_choice all \
--gpu 0
📊 5. Outputs
Model Checkpoints

Saved every 50 epochs in:

model/ckpt/

Includes:

generator_xxx.pt

discriminator_xxx.pt

Generated Results

Every 100 epochs:

5000 generated defect point clouds

Saved in:

model/generated/

These can be used for:

Offline quantitative evaluation

Structural morphology analysis

Statistical modeling

🧠 Notes

Designed for structural pipeline defect modeling.

Easily adaptable to other 3D point cloud datasets.

Lightweight implementation without heavy external frameworks.
