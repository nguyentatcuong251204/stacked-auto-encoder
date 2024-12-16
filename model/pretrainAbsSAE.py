import torch
import torch.nn as nn
import torch.nn.functional as F


class AbsSAE(nn.Module):
    def __init__(self, encoder = None, decoder = None,checkpoint = None,
                  is_variant = False):
        super(AbsSAE, self).__init__()
        self.checkpoint = checkpoint
        self.encoder = encoder
        self.decoder = decoder
        self.model = nn.Sequential(
            encoder,
            decoder
        )

        self.is_variant = is_variant

    def getEncoder(self):
        return self.encoder
    
    def getDecoder(self):
        return self.decoder
    

    def forward(self, x):
        

        if self.is_variant:
            
            out, mean, logvar = self.model(x)
            return out, mean, logvar
        
        
        return self.model(x)


    def make_soil(self,tensor_image):
        noise = torch.randint(0,10,tensor_image.shape) * 0.1
        noise = noise.to('cuda')
        for i in range(2):
            tensor_image = tensor_image + noise
            tensor_image = F.avg_pool2d(tensor_image, kernel_size=3, stride=1, padding=(3-1)//2)
        return tensor_image

    def save_checkpoint(self):
        if self.is_variant:
            torch.save(self.encoder.state_dict(), self.checkpoint[0])
            torch.save(self.decoder.state_dict(), self.checkpoint[1])
        else:
            n = len(self.checkpoint)
            for i in range(n-1):
                torch.save(self.encoder.global_encoder[i].state_dict(),self.checkpoint[i])
            
            torch.save(self.decoder.state_dict(), self.checkpoint[n-1])
    
    def inference(self, x):
        if self.is_variant == False:
            n = len(self.checkpoint)
            for i in range(n-1):
                self.encoder.global_encoder[i].load_state_dict(torch.load(self.checkpoint[i]))
                x = self.encoder.global_encoder[i](x)
            
            self.decoder.load_state_dict(torch.load(self.checkpoint[n-1]))
            x = self.decoder(x)
            return x
        
        self.encoder.load_state_dict(torch.load(self.checkpoint[0]))
        self.decoder.load_state_dict(torch.load(self.checkpoint[1]))
        x = self.model(x)
        return x
    
            
        
