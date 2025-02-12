import torch
import torch.nn as nn

class TransformerNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Bloco inicial (não altera o tamanho)
        self.initial = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4),
            nn.InstanceNorm2d(32),
            nn.ReLU()
        )
        
        # Blocos residuais (não alteram o tamanho)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(32) for _ in range(5)]
        )
        
        # Bloco de upsampling (garantindo saída 256x256)
        self.upsample = nn.Sequential(
            # Upsample 1 (256x256 → 256x256)
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            
            # Upsample 2 (256x256 → 256x256)
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            
            # Camada final (sem upsampling)
            nn.Conv2d(8, 3, kernel_size=9, stride=1, padding=4),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.residual_blocks(x)
        x = self.upsample(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)