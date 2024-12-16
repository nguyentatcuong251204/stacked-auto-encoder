from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import torch.nn as nn
import os

class CCCPDDataset(Dataset):
    def __init__(self, HR_folder_path, LR_folder_path):
        super().__init__()
        self.HR_folder_path = HR_folder_path
        self.LR_folder_path = LR_folder_path
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256,256))
        ])
    def __len__(self):
        if len(os.listdir(self.HR_folder_path)) != len(os.listdir(self.LR_folder_path)):
            raise ValueError("HR image number is not same LR image number!")

        return len(os.listdir(self.HR_folder_path))

    def __getitem__(self, index):
        HR_image_path = os.path.join(self.HR_folder_path,os.listdir(self.HR_folder_path)[index])
        LR_image_path = os.path.join(self.LR_folder_path,os.listdir(self.LR_folder_path)[index])
        HR_image = Image.open(HR_image_path)
        HR_image = self.transforms(HR_image)
        LR_image = Image.open(LR_image_path)
        LR_image = self.transforms(LR_image)
        return HR_image, LR_image

HR_folder_path = r"D:\Paper\ccpd_np_blur_3021"
LR_folder_path = r"D:\Paper\ccpd_np_clean_3021"

dataset = CCCPDDataset(HR_folder_path, LR_folder_path)
