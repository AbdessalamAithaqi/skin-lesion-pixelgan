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
project_root/
│── .venv/                   # Virtual environment (Python dependencies)
│── data/                     # Data directory
│   │── input_images/        # Your start images
│   │── test/                # Test dataset (generated)
│   │── train/               # Training dataset (generated)
│── pixelgan_models/         # Saved model checkpoints
│   │── 20250320_223547/     # Model version (generated)
│── pixelgan_outputs/        # Generated images/output results
│   │── 20250320_223547/     # Outputs for model version (generated)
│   │── generated_final/     # Final output images (generated)
│── .gitignore               # Git ignore file
│── dataset.py               # Dataset processing script
│── generate.py              # Image generation script (inference)
│── LICENSE                  # License file
│── models.py                # Defines the GAN model
│── preprocess.py            # Preprocessing script for images
│── README.md                # Documentation
│── requirements.txt         # Dependencies
│── train.py                 # Training script
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
python preprocess.py --source_dir .\data\input_images\ --data_dir data
```

### Training

Train the PixelGAN model:
```bash
python train.py --data_dir data --batch_size 16 --amp --n_epochs 100
```

### Generating Synthetic Images

Generate synthetic images using the trained model:
```bash
python generate.py --model_folder pixelgan_models\INSERT_MODEL_VERSION --input_dir data/test --num_images 20 --output_dir pixelgan_outputs/generated_final
```

## Results

Example results comparing original images (left), masked images (middle), and generated images (right):

[Results will be shown here]

## Academic Context

This project was developed as part of a coursework assignment for [COMP425] at [Concordia University]. The goal was to implement a GAN architecture to generate synthetic medical images and evaluate its effectiveness in improving downstream AI model performance.

## Contributors

- [Aymane Aaquil](https://github.com/aymaneaaquil)
- [Abdessalam Ait Haqi](https://github.com/AbdessalamAithaqi)
- [Alexander Chneerov](https://github.com/achneerov)
- [Cameron Shortt](https://github.com/cameronsshortt)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The original Pix2Pix paper: [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)
- [PyTorch](https://pytorch.org/) team for the deep learning framework
