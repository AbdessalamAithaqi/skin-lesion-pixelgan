import os
import time
import argparse
import datetime
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models import Generator, Discriminator
from dataset import get_data_loaders


def train(args):
    # Create output directories
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("pixelgan_outputs", timestamp)
    model_dir = os.path.join("pixelgan_models", timestamp)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize models
    generator = Generator(in_channels=3, out_channels=3).to(device)
    discriminator = Discriminator(in_channels=6).to(device)
    
    # Initialize optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    
    # Loss functions
    criterion_GAN = nn.MSELoss()
    criterion_pixelwise = nn.L1Loss()
    
    # Weight for pixel-wise loss
    lambda_pixel = 100
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers
    )
    
    # Training loop
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(args.n_epochs):
        for i, batch in enumerate(train_loader):
            # Get batch data
            real_A = batch['masked'].to(device)  # Input (masked)
            real_B = batch['original'].to(device)  # Target (original)
            
            # Adversarial ground truths
            valid = torch.ones((real_A.size(0), 1, 16, 16), requires_grad=False).to(device)
            fake = torch.zeros((real_A.size(0), 1, 16, 16), requires_grad=False).to(device)
            
            # ------------------
            #  Train Generator
            # ------------------
            optimizer_G.zero_grad()
            
            # Generate fake image
            fake_B = generator(real_A)
            
            # GAN loss
            pred_fake = discriminator(real_A, fake_B)
            loss_GAN = criterion_GAN(pred_fake, valid)
            
            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(fake_B, real_B)
            
            # Total generator loss
            loss_G = loss_GAN + lambda_pixel * loss_pixel
            
            loss_G.backward()
            optimizer_G.step()
            
            # --------------------
            #  Train Discriminator
            # --------------------
            optimizer_D.zero_grad()
            
            # Real loss
            pred_real = discriminator(real_A, real_B)
            loss_real = criterion_GAN(pred_real, valid)
            
            # Fake loss
            pred_fake = discriminator(real_A, fake_B.detach())
            loss_fake = criterion_GAN(pred_fake, fake)
            
            # Total discriminator loss
            loss_D = (loss_real + loss_fake) * 0.5
            
            loss_D.backward()
            optimizer_D.step()
            
            # Print progress
            if i % args.print_freq == 0:
                elapsed_time = time.time() - start_time
                print(f"[Epoch {epoch}/{args.n_epochs}] [Batch {i}/{len(train_loader)}] "
                      f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}, pixel: {loss_pixel.item():.4f}, adv: {loss_GAN.item():.4f}] "
                      f"Time: {elapsed_time:.2f}s")
            
            # Save sample images
            if i % args.sample_freq == 0:
                img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
                save_image(img_sample, os.path.join(output_dir, "samples", f"epoch_{epoch}_batch_{i}.png"),
                           nrow=args.batch_size, normalize=True)
        
        # Save models periodically
        if (epoch + 1) % args.save_freq == 0 or epoch == args.n_epochs - 1:
            torch.save(generator.state_dict(), os.path.join(model_dir, f"generator_epoch_{epoch+1}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(model_dir, f"discriminator_epoch_{epoch+1}.pth"))
            print(f"Models saved at epoch {epoch+1}")
    
    print(f"Training completed in {time.time() - start_time:.2f}s")
    print(f"Generated samples saved in {output_dir}/samples")
    print(f"Model checkpoints saved in {model_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train PixelGAN for skin lesion synthesis')
    
    # Data and output directories
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing the dataset')
    
    # Training parameters
    parser.add_argument('--n_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 for Adam optimizer')
    parser.add_argument('--img_size', type=int, default=256, help='Size of input/output images')
    
    # Hardware options
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA usage')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for data loading')
    
    # Logging and saving
    parser.add_argument('--print_freq', type=int, default=100, help='Frequency for printing training statistics')
    parser.add_argument('--sample_freq', type=int, default=200, help='Frequency for saving sample images')
    parser.add_argument('--save_freq', type=int, default=10, help='Frequency for saving model checkpoints (epochs)')
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()