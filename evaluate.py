import os
import torch
import numpy as np
import pandas as pd
from model.pretrainAbsSAE import AbsSAE
from config import experiment_config
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from dataset import CCCPDDataset
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as ssim

device = experiment_config["device"]
clean_folder_path = experiment_config["clean_folder_path"]
soil_folder_path = experiment_config["soil_folder_path"]

test_dataset = CCCPDDataset(clean_folder_path, soil_folder_path)
testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


def calculate_psnr(original, compressed):
    # Đảm bảo hai hình ảnh có cùng kích thước
    assert original.shape == compressed.shape, "Hình ảnh phải có cùng kích thước"

    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')  # PSNR vô cùng nếu không có sai khác

    max_pixel = 1  # Giả sử hình ảnh có độ sâu 8-bit
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr



def calculate_ssim(img1, img2):
    
    if img1.shape != img2.shape:
        raise ValueError("Hai ảnh phải có cùng kích thước.")

    # Tách các kênh màu (RGB)
    ssim_values = []
    for i in range(3):  # Duyệt qua 3 kênh màu
        ssim_value = ssim(img1[:, :, i], img2[:, :, i], data_range=img2[:, :, i].max() - img2[:, :, i].min())
        ssim_values.append(ssim_value)

    # Tính SSIM trung bình trên 3 kênh
    return np.mean(ssim_values)


def calculate_nmse(img1, img2):
    
    # Kiểm tra kích thước ảnh
    if img1.shape != img2.shape:
        raise ValueError("Hai ảnh phải có cùng kích thước.")

    # Tính NMSE
    numerator = np.sum((img1 - img2) ** 2)
    denominator = np.sum(img1 ** 2)
    nmse_value = numerator / denominator if denominator != 0 else float('inf')
    return nmse_value


for model_name, model_config in experiment_config["Model"].items():
    torch.cuda.empty_cache()
    print(model_name)
    model = AbsSAE(encoder=model_config["encoder"], decoder=model_config["decoder"],
                   checkpoint=model_config["checkpoint"], is_variant=model_config["is_variant"]).to(device)
    
    i = 0
    psnr_score = []
    ssim_score = []
    nmse_score = []
    dic = {"METHOD":[], "PSNR": None, "SSIM": None, "NMSE": None}
    for HR, LR in testloader:
        model.eval()
        HR, LR = HR.to(device), LR.to(device)
        if model.is_variant == False:
            HR_output = model.inference(LR)
        else:
            HR_output = model.inference(LR)[0]
            HR_output = torch.sigmoid(HR_output)
        HR_output = HR_output.squeeze(0).permute(1,2,0).detach().cpu().numpy()
        HR = HR.squeeze(0).permute(1,2,0).detach().cpu().numpy()

        psnr_score.append(calculate_psnr(HR, HR_output))
        ssim_score.append(calculate_ssim(HR, HR_output))
        nmse_score.append(calculate_nmse(HR, HR_output))

        dic["METHOD"].append(model_name)

    dic["PSNR"] = psnr_score
    dic["SSIM"] = ssim_score
    dic["NMSE"] = nmse_score
    dataframe = pd.DataFrame(dic)
    dataframe.to_csv(f"{model_name}_resnet50_pretrain_noise.csv")
