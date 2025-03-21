import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm


class UNetDown(nn.Module):
    """
    Downsampling block for the UNet-based generator
    """
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0, use_spectral_norm=False):
        super(UNetDown, self).__init__()
        
        if use_spectral_norm:
            layers = [spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False))]
        else:
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        
        layers.append(nn.LeakyReLU(0.2, inplace=True))  # inplace=True for memory efficiency
        
        if dropout:
            layers.append(nn.Dropout(dropout))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """
    Upsampling block for the UNet-based generator
    """
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(UNetUp, self).__init__()
        
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)  # inplace=True for memory efficiency
        ]
        
        if dropout:
            layers.append(nn.Dropout(dropout))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class Generator(nn.Module):
    """
    Generator model based on U-Net architecture from Pix2Pix
    Optimized for GPU performance
    """
    def __init__(self, in_channels=3, out_channels=3, use_spectral_norm=False):
        super(Generator, self).__init__()
        
        # Initial downsampling
        self.down1 = UNetDown(in_channels, 64, normalize=False, use_spectral_norm=use_spectral_norm)  # 128x128
        self.down2 = UNetDown(64, 128, use_spectral_norm=use_spectral_norm)  # 64x64
        self.down3 = UNetDown(128, 256, use_spectral_norm=use_spectral_norm)  # 32x32
        self.down4 = UNetDown(256, 512, dropout=0.5, use_spectral_norm=use_spectral_norm)  # 16x16
        self.down5 = UNetDown(512, 512, dropout=0.5, use_spectral_norm=use_spectral_norm)  # 8x8
        self.down6 = UNetDown(512, 512, dropout=0.5, use_spectral_norm=use_spectral_norm)  # 4x4
        self.down7 = UNetDown(512, 512, dropout=0.5, use_spectral_norm=use_spectral_norm)  # 2x2
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5, use_spectral_norm=use_spectral_norm)  # 1x1
        
        # Upsampling
        self.up1 = UNetUp(512, 512, dropout=0.5)  # 2x2
        self.up2 = UNetUp(1024, 512, dropout=0.5)  # 4x4
        self.up3 = UNetUp(1024, 512, dropout=0.5)  # 8x8
        self.up4 = UNetUp(1024, 512, dropout=0.5)  # 16x16
        self.up5 = UNetUp(1024, 256)  # 32x32
        self.up6 = UNetUp(512, 128)  # 64x64
        self.up7 = UNetUp(256, 64)  # 128x128
        
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
        
        # Upsampling
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        
        return self.final(u7)


class Discriminator(nn.Module):
    """
    Discriminator model for the PixelGAN (PatchGAN)
    Optimized for GPU performance with spectral normalization for stability
    """
    def __init__(self, in_channels=6, use_spectral_norm=True):  # 3 channels for input + 3 channels for target
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_channels, out_channels, normalize=True):
            if use_spectral_norm:
                conv = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
            else:
                conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
                
            layers = [conv]
            
            if normalize and not use_spectral_norm:
                layers.append(nn.InstanceNorm2d(out_channels))
                
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        # Input: concatenated input and target images
        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),  # 128x128
            *discriminator_block(64, 128),  # 64x64
            *discriminator_block(128, 256),  # 32x32
            *discriminator_block(256, 512),  # 16x16
            nn.ZeroPad2d((1, 0, 1, 0)),
            spectral_norm(nn.Conv2d(512, 1, kernel_size=4, padding=1)) if use_spectral_norm else 
            nn.Conv2d(512, 1, kernel_size=4, padding=1)  # 16x16 patches
        )
        
        # Initialize weights for better convergence
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) and not isinstance(m, nn.utils.spectral_norm.SpectralNorm):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    
    def forward(self, img_A, img_B):
        # Concatenate input and output image by channels
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)