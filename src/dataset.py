import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms

class FERDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pixels = np.array(row["pixels"].split(), dtype=np.uint8) 
        image = pixels.reshape(48, 48)
        
        if self.transform:
            image = transforms.ToPILImage()(image)
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0
            
        label = torch.tensor(row["emotion"], dtype=torch.long)
        return image, label

def get_dataloaders(df, batch_size=64, test_size=0.2, random_state=42):
    train_data, val_data = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["emotion"]
    )
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(), 
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_loader = DataLoader(FERDataset(train_data, transform=train_transform), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(FERDataset(val_data, transform=val_transform), batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader