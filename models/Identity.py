import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels//8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//8, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        att = self.attention(x)
        out = att * x
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        return out, att

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        return out

class IdentityAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc_block1 = EncoderBlock(3, 32)
        self.enc_block2 = EncoderBlock(32, 64)
        self.enc_block3 = EncoderBlock(64, 128)
        self.enc_block4 = EncoderBlock(128, 256)

        # Decoder
        self.dec_block1 = DecoderBlock(256, 128)
        self.dec_block2 = DecoderBlock(128+128, 64)
        self.dec_block3 = DecoderBlock(64+64, 32)
        self.dec_block4 = DecoderBlock(32+32, 3)
        
    def forward(self, x):
        # Encoder
        x1, att1 = self.enc_block1(x)
        x2, att2 = self.enc_block2(x1)
        x3, att3 = self.enc_block3(x2)
        x4, att4 = self.enc_block4(x3)
        # Decoder
        u = self.dec_block1(x4)
        u = torch.cat((u, x3), dim=1)
        u = self.dec_block2(u)
        u = torch.cat((u, x2), dim=1)
        u = self.dec_block3(u)
        u = torch.cat((u, x1), dim=1)
        u = self.dec_block4(u)
        
        return u
    
if __name__ == '__main__':
    AE = IdentityAE()
    print(AE(torch.ones([1, 3, 384, 512])).shape)