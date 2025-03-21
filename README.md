# Skin Lesion PixelGAN

A PyTorch implementation of PixelGAN/Pix2Pix for generating synthetic skin lesion images to enhance cancer detection AI training datasets.

## Overview

This project implements a PixelGAN (based on Pix2Pix architecture) to generate realistic synthetic skin lesion images. It's designed to augment training data for machine learning models that detect and classify skin cancer.

### Core Features
- Generate high-quality synthetic skin lesion images
- Train on limited datasets and expand them through GAN-based synthesis
- Evaluate synthetic image quality through medical professional assessment
- Improve downstream cancer detection AI performance
- Leverages GPU acceleration (CUDA) for faster training and inference

## Project Structure

```
skin-lesion-pixelgan/
├── data/               # Data directory (not tracked by git)
│   ├── train/          # Training images
│   └── test/           # Test images
├── pixelgan_models/    # Saved model checkpoints
├── pixelgan_outputs/   # Generated images
├── models.py           # Model architecture definitions
├── dataset.py          # Dataset loading and preprocessing
├── train.py            # Training script
├── generate.py         # Image generation script
└── preprocess.py       # Data preprocessing utilities
```

## Requirements

- Python 3.8+
- PyTorch 2.5.1+ (CUDA 12.1)
- opencv-python (for image processing)
- CUDA-enabled GPU (NVIDIA RTX recommended)
- torchvision
- Pillow
- numpy

You can install the required packages using:
```bash
pip install -r requirements.txt
```

### Leveraging GPU for Faster Training
To install the GPU-accelerated version of PyTorch with CUDA 12.1 support, use:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

To ensure PyTorch is using your GPU, run:
```python
import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))  # Should show your GPU name
```
If CUDA is not available, ensure you have installed the correct PyTorch version for your CUDA setup.

## Usage

### Data Preparation

1. Place your skin lesion images in the `data/` directory
2. Preprocess images:
```bash
python preprocess.py --source_dir source_images --data_dir data
```

### Training

Train the PixelGAN model:
```bash
python train.py --data_dir data --n_epochs 200 --batch_size 8
```

### Generating Synthetic Images

Generate synthetic images using the trained model:
```bash
python generate.py --model_folder MODEL_TIMESTAMP --input_dir data/test --num_images 50
```

## Results

Example results comparing original images (left), masked images (middle), and generated images (right):

[Results will be shown here]

## Academic Context

This project was developed as part of a coursework assignment for [COMP425] at [Concordia University]. The goal was to implement a GAN architecture to generate synthetic medical images and evaluate its effectiveness in improving downstream AI model performance.

## Contributors

- Aymane Aaquil
- Abdessalam Ait Haqi
- Alex Chneerov
- Cameron

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The original Pix2Pix paper: [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)
- [PyTorch](https://pytorch.org/) team for the deep learning framework
