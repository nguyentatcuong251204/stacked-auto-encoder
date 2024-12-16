import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
resnet50 = models.resnet50(pretrained=True)

class Encoder(nn.Module):
    def __init__(self, n_block = 1, previous_checkpoint = None):
        super().__init__()
        self.previous_checkpoint = previous_checkpoint
        self.encoder1 = nn.Sequential(
                resnet50.conv1,
                resnet50.bn1,
                resnet50.relu,
                resnet50.maxpool
                )
        
        self.encoder2 = resnet50.layer1
        self.encoder3 = resnet50.layer2
        self.encoder4 = resnet50.layer3
        

        if n_block == 1:
            self.global_encoder = [self.encoder1]

        elif n_block == 2:
            self.global_encoder = [self.encoder1, self.encoder2]

        elif n_block == 3:
            self.global_encoder = [self.encoder1, self.encoder2, self.encoder3]

        else:
            self.global_encoder = [self.encoder1, self.encoder2, self.encoder3, self.encoder4]

    
    def forward(self, x):
        if self.previous_checkpoint:
            for i in range(len(self.previous_checkpoint)):
                self.global_encoder[i].load_state_dict(torch.load(self.previous_checkpoint[i]))
                x = self.make_soil(x)
                x = self.global_encoder[i](x)
            return self.global_encoder[-1](x)

        
        for encoder in self.global_encoder:
            x = encoder(x)
        return x
    
    def make_soil(self,tensor_image):
        noise = torch.randint(0,10,tensor_image.shape) * 0.1
        noise = noise.to('cuda')
        for i in range(1):
            tensor_image = tensor_image + noise
            tensor_image = F.avg_pool2d(tensor_image, kernel_size=3, stride=1, padding=(3-1)//2)
        return tensor_image
        


class Decoder(nn.Module):
    def __init__(self, n_block = 4, size = 256):
        super().__init__()
        self.size = size
        if n_block == 1:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(64 , 3, kernel_size=3, stride=2, padding=1),
            )
        elif n_block == 2:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(256 , 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 3, kernel_size=3 , stride=2, padding=1)
            )
        elif n_block == 3:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1)
            )
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1)
            )

    def forward(self, x):
        out = self.up(x)
        return TF.resize(out,(self.size, self.size))
    
