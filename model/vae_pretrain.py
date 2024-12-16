
import torch
import math
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as TF
import torch.nn.functional as F
# Tải mô hình ResNet50 đã được tiền huấn luyện
resnet50 = models.resnet50(pretrained=True)

class Encoder(nn.Module):
    def __init__(self, size = 256, n_dims_out = 128):
        super(Encoder, self).__init__()
        
        self.conv_block = nn.Sequential(
        resnet50.conv1,
        resnet50.bn1,
        resnet50.relu,
        resnet50.maxpool,

        resnet50.layer1,
        resnet50.layer2,
        resnet50.layer3
        )

        x = self.conv_block(torch.rand(1,3,size,size))
        shape = x.shape

        self.fc1 = nn.Linear(shape[1]*shape[2]*shape[3], n_dims_out)  # Latent space mean
        self.fc2 = nn.Linear(shape[1]*shape[2]*shape[3], n_dims_out)  # Latent space log variance
        
    def forward(self,x):
        x = self.conv_block(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        # Latent space
        z_mean = self.fc1(x)
        z_log_var = self.fc2(x)
        z = self.reparameterize(z_mean, z_log_var)
        return z_mean, z_log_var, z
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

class Decoder(nn.Module):
    def __init__(self, size = 256, n_dims_out = 128):
        super(Decoder, self).__init__()
        self.size_input = size

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(1024 , 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
            )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512 , 256 , kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256 , 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.up4 = nn.ConvTranspose2d(64 , 3, kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Linear(n_dims_out, 1024*16*16)
        
    def forward(self, z):
        # Fully connected to reshape latent space
        x = F.relu(self.fc1(z[2]))
        x = x.view(x.size(0), 1024, 16, 16)  # Reshape into feature map
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return torch.sigmoid(TF.resize(x,(self.size_input,self.size_input))), z[0], z[1]
