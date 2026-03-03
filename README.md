A Multi-Strategy Fusion-Based Approach for Pipeline Defect Segmentation and Quantification

This repository contains the official PyTorch implementation of the paper:

"A Multi-Strategy Fusion-Based Approach for Pipeline Defect Segmentation and Quantification"

The project provides a lightweight and pure PyTorch framework for 3D pipeline defect learning, designed for easy integration with custom datasets.

🔧 1. Environment Preparation

Please ensure your system satisfies the following requirements:

Python 3.x

PyTorch ≥ 1.4 (CUDA recommended)

NumPy

Visdom (for real-time visualization)

Install Dependencies
pip install torch torchvision numpy visdom

A CUDA-enabled GPU is highly recommended for training.

📂 2. Custom Dataset Preparation

The project uses a custom data loader (data_benchmark.py) that directly supports .npy and .txt point cloud files.

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

Each file must contain 3D point cloud coordinates: (X, Y, Z)

The loader will:

Automatically sample to 2048 points

Normalize data during training

⚙️ 3. Configuration Settings

All hyperparameters and path configurations are managed in:

arguments.py

You may edit the file directly or override parameters via the command line.

Important Arguments
Argument	Description	Default
--dataset_path	Absolute path to dataset (Required)	None
--class_choice	Specific defect folder or all	all
--point_num	Number of sampled points	2048
--batch_size	Training batch size	32
--epochs	Total training epochs	2000
--gpu	GPU index	0
🚀 4. Training the Model
Step 1: Start Visdom Server

Open a new terminal:

python -m visdom.server -port 8097

Then open your browser:

http://localhost:8097

This dashboard shows:

Generator loss

Discriminator loss

3D point cloud visualization

Step 2: Launch Training

In your main terminal:

python train.py \
--dataset_path /absolute/path/to/your/dataset \
--class_choice all \
--gpu 0
📊 5. Outputs and Checkpoints
Model Checkpoints

Saved every 50 epochs

Location:

model/ckpt/

Files:

generator_xxx.pt

discriminator_xxx.pt

Generated Results

Generated every 100 epochs

5000 synthetic defect point clouds per batch

Saved to:

model/generated/

These files can be used for:

Offline quantitative evaluation

Structural analysis

Statistical defect distribution modeling

🧠 Notes

Designed for structural pipeline defect learning.

Easily adaptable to other 3D point cloud datasets.

Lightweight implementation without heavy framework dependencies.
