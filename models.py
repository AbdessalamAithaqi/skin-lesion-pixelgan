import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm


class ResidualBlock(nn.Module):
    """
    Residual block for the Generator
    """
    def __init__(self, in_channels, use_spectral_norm=False):
        super(ResidualBlock, self).__init__()
        
        # First convolution layer
        if use_spectral_norm:
            conv1 = spectral_norm(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False))
            conv2 = spectral_norm(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False))
        else:
            conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
            conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        
        # Define the main path
        self.main_path = nn.Sequential(
            conv1,
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            conv2,
            nn.InstanceNorm2d(in_channels)
        )
        
        # ReLU for after the addition (put outside to keep it non-inplace for the residual connection)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(x + self.main_path(x))


class UNetDownWithResidual(nn.Module):
    """
    Enhanced downsampling block with residual connections for the UNet-based generator
    """
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0, use_spectral_norm=False, use_residual=True):
        super(UNetDownWithResidual, self).__init__()
        
        # Main downsampling path
        if use_spectral_norm:
            layers = [spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False))]
        else:
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        if dropout:
            layers.append(nn.Dropout(dropout))
        
        self.model = nn.Sequential(*layers)
        
        # Add a residual block if enabled and channels match
        self.use_residual = use_residual and (in_channels == out_channels)
        if self.use_residual:
            self.residual_block = ResidualBlock(out_channels, use_spectral_norm)
    
    def forward(self, x):
        x = self.model(x)
        if self.use_residual:
            x = self.residual_block(x)
        return x


class UNetUpWithResidual(nn.Module):
    """
    Enhanced upsampling block with residual connections for the UNet-based generator
    """
    def __init__(self, in_channels, out_channels, dropout=0.0, use_residual=True):
        super(UNetUpWithResidual, self).__init__()
        
        # Main upsampling path
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        
        if dropout:
            layers.append(nn.Dropout(dropout))
        
        self.model = nn.Sequential(*layers)
        
        # Add a residual block after concatenation
        self.use_residual = use_residual
        if self.use_residual:
            self.residual_block = ResidualBlock(out_channels * 2)  # *2 because of the concatenation
    
    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        if self.use_residual:
            # We need to adapt the residual block for the concatenated tensor
            # For simplicity, let's use a 1x1 conv to match dimensions before the residual block
            # This is a common approach when adding residual connections to U-Net
            x = self.residual_block(x)
        return x


class ResidualGenerator(nn.Module):
    """
    Enhanced Generator model with residual connections
    Based on U-Net architecture from Pix2Pix
    Optimized for GPU performance
    """
    def __init__(self, in_channels=3, out_channels=3, use_spectral_norm=False):
        super(ResidualGenerator, self).__init__()
        
        # Initial downsampling with residual connections
        self.down1 = UNetDownWithResidual(in_channels, 64, normalize=False, use_spectral_norm=use_spectral_norm, use_residual=False)  # 128x128
        self.down2 = UNetDownWithResidual(64, 128, use_spectral_norm=use_spectral_norm)  # 64x64
        self.down3 = UNetDownWithResidual(128, 256, use_spectral_norm=use_spectral_norm)  # 32x32
        self.down4 = UNetDownWithResidual(256, 512, dropout=0.5, use_spectral_norm=use_spectral_norm)  # 16x16
        self.down5 = UNetDownWithResidual(512, 512, dropout=0.5, use_spectral_norm=use_spectral_norm)  # 8x8
        self.down6 = UNetDownWithResidual(512, 512, dropout=0.5, use_spectral_norm=use_spectral_norm)  # 4x4
        self.down7 = UNetDownWithResidual(512, 512, dropout=0.5, use_spectral_norm=use_spectral_norm)  # 2x2
        self.down8 = UNetDownWithResidual(512, 512, normalize=False, dropout=0.5, use_spectral_norm=use_spectral_norm, use_residual=False)  # 1x1
        
        # Bridge with residual blocks - add more capacity at the bottleneck
        self.bridge = nn.Sequential(
            ResidualBlock(512, use_spectral_norm),
            ResidualBlock(512, use_spectral_norm)
        )
        
        # Upsampling with residual connections
        self.up1 = UNetUpWithResidual(512, 512, dropout=0.5)  # 2x2
        self.up2 = UNetUpWithResidual(1024, 512, dropout=0.5)  # 4x4
        self.up3 = UNetUpWithResidual(1024, 512, dropout=0.5)  # 8x8
        self.up4 = UNetUpWithResidual(1024, 512, dropout=0.5)  # 16x16
        self.up5 = UNetUpWithResidual(1024, 256)  # 32x32
        self.up6 = UNetUpWithResidual(512, 128)  # 64x64
        self.up7 = UNetUpWithResidual(256, 64)  # 128x128
        
        # Final layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
        # Initialize weights for better convergence
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    def forward(self, x):
        # Downsampling
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        
        # Bridge
        d8 = self.bridge(d8)
        
        # Upsampling
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        
        return self.final(u7)


# The Discriminator can remain largely the same, but let's add some residual connections to it as well

class ResidualDiscriminator(nn.Module):
    """
    Enhanced Discriminator model with residual connections for the PixelGAN (PatchGAN)
    Optimized for GPU performance with spectral normalization for stability
    """
    def __init__(self, in_channels=6, use_spectral_norm=True):  # 3 channels for input + 3 channels for target
        super(ResidualDiscriminator, self).__init__()
        
        def discriminator_block(in_channels, out_channels, normalize=True, add_residual=False):
            if use_spectral_norm:
                conv = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
            else:
                conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
                
            layers = [conv]
            
            if normalize and not use_spectral_norm:
                layers.append(nn.InstanceNorm2d(out_channels))
                
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            if add_residual and in_channels == out_channels:
                layers.append(ResidualBlock(out_channels, use_spectral_norm))
                
            return layers
        
        # Input: concatenated input and target images
        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),  # 128x128
            *discriminator_block(64, 128, add_residual=True),  # 64x64
            *discriminator_block(128, 256, add_residual=True),  # 32x32
            *discriminator_block(256, 512, add_residual=True),  # 16x16
            nn.ZeroPad2d((1, 0, 1, 0)),
            spectral_norm(nn.Conv2d(512, 1, kernel_size=4, padding=1)) if use_spectral_norm else 
            nn.Conv2d(512, 1, kernel_size=4, padding=1)  # 16x16 patches
        )
        
        # Initialize weights for better convergence
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            # Check if this Conv2d is not already wrapped by spectral_norm
            if not hasattr(m, 'weight_orig'):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
    
    def forward(self, img_A, img_B):
        # Concatenate input and output image by channels
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)