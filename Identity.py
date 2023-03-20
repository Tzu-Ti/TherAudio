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
            nn.Conv2d(out_channels, out_channels//8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//8, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        att = self.attention(out)
        out = att*out
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        return out, att

class IdentityAE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, heads, dropout):
        super().__init__()
        self.w, self.h = input_size
        
        # Encoder layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * (self.w // 8) * (self.h // 8), hidden_size)
        
        # Transformer layers
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        # Decoder layers
        self.fc2 = nn.Linear(hidden_size, 128 * (self.w // 8) * (self.h // 8))
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        
        # Transformer
        x = x.unsqueeze(0)
        x = self.transformer_encoder(x)
        x = x.squeeze(0)
        
        # Decoder
        x = F.relu(self.fc2(x))
        x = x.view(-1, 128, (self.w // 8), (self.h // 8))
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.deconv3(x)
        
        return x
    
if __name__ == '__main__':
    AE = IdentityAE(input_size=[384, 512], hidden_size=256, num_layers=2, heads=8, dropout=0.1)
    print(AE(torch.ones([1, 3, 384, 512])).shape)