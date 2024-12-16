from tqdm import tqdm
import torch
from model.pretrainAbsSAE import AbsSAE
from config import experiment_config
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam
import torch.nn as nn
from dataset import CCCPDDataset
# from sklearn.model_selection import train_test_split

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
clean_folder_path = experiment_config["clean_folder_path"]
soil_folder_path = experiment_config["soil_folder_path"]
dataset = CCCPDDataset(clean_folder_path, soil_folder_path)

batch_size = experiment_config["batch_size"]
epoch = experiment_config["epoch"]
lr = experiment_config["lr"]
loss_fn = nn.MSELoss()

# train, test = train_test_split(dataset, test_size=0.2, random_state=42)
trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
# testloader = DataLoader(test, batch_size=2, shuffle=False)


def loss_function(x_recon, x, mu, logvar):
    # reconstruction loss
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')

    # latent loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + 0.1*KLD



torch.cuda.empty_cache()

for model_name, model_config in experiment_config["Model"].items():
    torch.cuda.empty_cache()
    print(model_name)
    model = AbsSAE(encoder=model_config["encoder"], decoder=model_config["decoder"], 
                   checkpoint=model_config["checkpoint"], is_variant=model_config["is_variant"]).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Tổng số tham số: {total_params}")
    optimizer = Adam(model.parameters(), lr=lr)
    
 
    for epoch in tqdm(range(experiment_config["epoch"]), desc=f"Training {model_name}", unit="epoch"):
        torch.cuda.empty_cache()
        model.train()
        sum_loss = 0
        for HR, LR in trainloader:
            torch.cuda.empty_cache()
            HR = HR.to(device)
            LR = LR.to(device)

        
            if model_config["is_variant"] == False:
                    optimizer.zero_grad()
                    HR_output = model(LR)
                    loss = loss_fn(HR_output, HR)
                    loss.backward()
                    sum_loss+=loss
                    optimizer.step()
                    
            else:
                    optimizer.zero_grad()
                    HR_output, muy, logvar = model(LR)
                    loss = loss_function(HR_output, HR, muy, logvar)
                    loss.backward()
                    sum_loss+=loss
                    optimizer.step()

        model.save_checkpoint()                                                 
        
        print(f"Model: {model_name}, epoch: {epoch}, loss: {sum_loss/len(trainloader)}")



    




