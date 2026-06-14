import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class FERDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pixels = np.array(row["pixels"].split(), dtype=np.float32)
        image = pixels.reshape(1, 48, 48) / 255.0  
        
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(row["emotion"], dtype=torch.long)
        return image, label

def get_dataloaders(df, batch_size=64, test_size=0.2, random_state=42):
    train_data, val_data = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["emotion"]
    )
    
    train_loader = DataLoader(FERDataset(train_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(FERDataset(val_data), batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader