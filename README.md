A Multi-Strategy Fusion-Based Approach for Pipeline Defect Segmentation and Quantification
This repository contains the official PyTorch codebase for the paper "A Multi-Strategy Fusion-Based Approach for Pipeline Defect Segmentation and Quantification".
This repository provides a lightweight, pure PyTorch implementation designed for easy integration with custom datasets.
Step 1: Environment Preparation
Ensure your system meets the basic computational requirements. A CUDA-enabled GPU is highly recommended for training.

Python 3.x

PyTorch (Version >= 1.4, CUDA supported)

Visdom (For real-time loss tracking and 3D point cloud visualization)

NumPy

Install the required dependencies using pip:

Bash
pip install torch torchvision numpy visdom
Step 2: Custom Dataset Preparation
This project uses a custom BenchmarkDataset loader (data_benchmark.py) capable of reading .npy and .txt files directly. Organize your 3D pipeline defect point clouds using the following directory structure:

Plaintext
/your_custom_dataset_path/
|-- Defect_Type_A/
|   |-- defect_001.npy 
|   |-- defect_002.txt
|-- Defect_Type_B/
|   |-- defect_001.npy
Note: Ensure your point clouds contain 3D coordinates (X, Y, Z). The data loader will automatically sample them to 2048 points and normalize them during the training loop.

Step 3: Configuration Settings
All core hyperparameters and path configurations are managed in arguments.py. Before starting the training process, verify and update the following arguments (you can also pass them via the command line):

--dataset_path: (Required) The absolute path to your custom dataset root directory.

--class_choice: The specific folder/defect type you want to train on (e.g., Defect_Type_A). Set this to all to train on the entire dataset.

--point_num: The number of points per sample (Default: 2048).

--batch_size: Batch size for training (Default: 32). Adjust based on your GPU VRAM.

--epochs: Total number of training epochs (Default: 2000).

Step 4: Training the Model
The training script utilizes Visdom to monitor the generator and discriminator loss curves, as well as to visualize the generated 3D defect structures in real-time.

1. Start the Visdom Server Open a new terminal window and start the visualization server:

Bash
python -m visdom.server -port 8097
(You can view the training dashboard by navigating to http://localhost:8097 in your web browser.)

2. Launch the Training Script Return to your primary terminal and run the training command, pointing to your dataset:

Bash
python train.py --dataset_path /absolute/path/to/your/dataset --class_choice all --gpu 0
Outputs & Checkpoints
Model Weights: By default, .pt checkpoint files for the Generator and Discriminator are saved every 50 epochs in the model/ckpt/ directory.

Generated Results: Every 100 epochs, the model generates a batch of 5000 defect point clouds and saves them as a tensor file in the model/generated/ directory for offline quantification and analysis.
