import torch
import torch.nn as nn
import numpy as np
import os, io, base64, gc
from PIL import Image

# Custom weights initialization
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

class PixelNorm(nn.Module):
    '''
    PixelNorm normalizes over channels which provides stability 
    over BatchNorm which relies on other images in the batch for 
    normalization.
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)

# Processes images into base64 for displaying 
def format_images(imageList):
    
    # Ensure it's a numpy array
    arr = np.array(imageList, dtype=np.uint8)
    
    # Convert to image
    img = Image.fromarray(arr)

    # Save into a memory buffer as PNG
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    
    # Encode to base64 for displaying
    return base64.b64encode(buffer.getvalue()).decode()

# Generator Class
class Generator(nn.Module):
    def __init__(self, latent_dim = 100):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024*1*1),  # [B, 1024]
            nn.LeakyReLU(0.2),
            
            nn.Linear(1024*1*1, 512*4*4),     # [B, 8192]
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (512, 4, 4))      # [B, 512, 4, 4]
        )
        self.convA = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(512, 256, 3, 1, 1),     # [B, 512, 8, 8] -> [B, 256, 8, 8]
            nn.LeakyReLU(0.2),
            PixelNorm(),
            
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, 1, 1),     # [B, 256, 16, 16] -> [B, 128, 16, 16]
            nn.LeakyReLU(0.2),
            PixelNorm(),
            
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, 1, 1),      # [B, 128, 32, 32] -> [B, 64, 32, 32]
            nn.LeakyReLU(0.2),
            PixelNorm(),
            
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, 1, 1),       # [B, 64, 64, 64] -> [B, 32, 64, 64]     
            nn.LeakyReLU(0.2),
            PixelNorm()
        )
        self.convB = nn.Sequential(

            nn.ConvTranspose2d(512, 256, 4, 2, 1),   # [B, 256, 8, 8]
            nn.LeakyReLU(0.2),
            PixelNorm(),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),   # [B, 128, 16, 16]
            nn.LeakyReLU(0.2),
            PixelNorm(),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),    # [B, 64, 32, 32]
            nn.LeakyReLU(0.2),
            PixelNorm(),
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1),     # [B, 32, 64, 64]
            nn.LeakyReLU(0.2),
            PixelNorm()
        )
        self.convOut = nn.Sequential(
            nn.Conv2d(64, 3, 1, 1, 0),               # [B, 3, 64, 64]
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        xA = self.convA(x)
        xB = self.convB(x)
        x = torch.cat((xA, xB), dim=1)
        return self.convOut(x)
    
# Critic Class
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.convA = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),      # [B, 64, 32, 32]
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 4, 2, 1),    # [B, 128, 16, 16]
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, 4, 2, 1),   # [B, 256, 8, 8]
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 512, 4, 2, 1),   # [B, 512, 4, 4]
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(512, 1024, 4, 2, 1),  # [B, 1024, 2, 2]
            nn.LeakyReLU(0.2),
            nn.Flatten()
        )
        self.convB = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),      # [B, 64, 64, 64]
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),             # [B, 64, 32, 32]
            
            nn.Conv2d(64, 128, 3, 1, 1),    # [B, 128, 32, 32]
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),             # [B, 256, 16, 16]
            
            nn.Conv2d(128, 256, 3, 1, 1),   # [B, 256, 16, 16]
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),             # [B, 256, 8, 8]
            
            nn.Conv2d(256, 512, 3, 1, 1),   # [B, 512, 8, 8]
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),             # [B, 512, 4, 4]
            
            nn.Conv2d(512, 1024, 3, 1, 1),  # [B, 1024, 4, 4]
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),             # [B, 1024, 2, 2]
            nn.Flatten()                    # [B, 4096]
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(2048*2*2, 1)          # [B, 1]
        )

    def forward(self, x):
        xA = self.convA(x)
        xB = self.convB(x)
        x = torch.cat((xA, xB), dim=1)
        return self.fc(x)
    
class ImageGenerator:
    def __init__(self, gen_file="gen_checkpoint_031.pth", critic_file="critic_checkpoint_031.pth"):

        # Create the model networks
        self.G = Generator()
        self.C = Critic()

        # Create the model save file paths
        gen_path = os.path.join("Models", gen_file)
        critic_path = os.path.join("Models", critic_file)

        # Load the parameters of the models
        self.G.load_state_dict(torch.load(gen_path, map_location=torch.device('cpu')))
        self.C.load_state_dict(torch.load(critic_path, map_location=torch.device('cpu')))

        # Set the models to evaluation mode
        self.G.eval()
        self.C.eval()
    
    def generate(self, latent_dim=100):

        with torch.no_grad():

            # Create multiples noises for the model        
            noise = torch.randn(100, latent_dim, dtype=torch.float32)

            # Generate the images and get their scores
            images = self.G(noise).cpu()
            scores = self.C(images).cpu()
            del noise

            # Get the image that has the low score of all (low score = better)
            idx = torch.argmin(scores).item()
            print(scores[idx].item())
            del scores

            # Convert the image into (H, W, C) format and turn it into numpy array for formatting
            image = images[idx].detach().permute(1, 2, 0).numpy()
            del images, idx

            # convert the image range from (-1, 1) to (0, 255)
            image = (((image + 1) / 2) * 255).astype(np.uint8)

            # Garbage collection
            gc.collect()

            # Convert the images to base64 to display
            return format_images(image)