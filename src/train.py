import torch
import numpy as np
import wandb
from src.models import LinearBaselineModel, SimpleCNN, SolidResNet, UltimateResNet, HybridViT

import numpy as np

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    alpha = 0.2 
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        if alpha > 0 and np.random.rand() > 0.5:
            lam = np.random.beta(alpha, alpha)
            batch_size = images.size(0)
            index = torch.randperm(batch_size).to(device)
            
            mixed_images = lam * images + (1 - lam) * images[index]
            labels_a, labels_b = labels, labels[index]
            
            optimizer.zero_grad()
            outputs = model(mixed_images)
            
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct_labels = labels_a if lam >= 0.5 else labels_b
            correct += predicted.eq(correct_labels).sum().item()
        else:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        running_loss += loss.item() * images.size(0)
        
    return running_loss / total, correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total

def validate_with_tta(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs_raw = model(images)
            
            flipped_images = torch.flip(images, dims=[-1])
            outputs_flipped = model(flipped_images)
            
            shifted_images = torch.roll(images, shifts=(2, 2), dims=(-2, -1))
            outputs_shifted = model(shifted_images)
            
            final_outputs = (outputs_raw + outputs_flipped + outputs_shifted) / 3.0
            
            loss = criterion(final_outputs, labels)
            running_loss += loss.item() * images.size(0)
            
            _, predicted = final_outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / total, correct / total

def run_experiment(config, train_loader, val_loader):
    wandb.init(
        project="FER2013-Project", 
        name=config.get("experiment_name", "nameless_run"), 
        config=config
    )
    cfg = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if cfg.model_name == "LinearBaseline":
        model = LinearBaselineModel().to(device)
    elif cfg.model_name == "SimpleCNN":
        model = SimpleCNN().to(device)
    elif cfg.model_name == "SolidResNet":
        model = SolidResNet().to(device)
    elif cfg.model_name == "UltimateResNet": 
        model = UltimateResNet().to(device)
    elif cfg.model_name == "HybridViT":
        model = HybridViT(
            num_classes=7,
            embed_dim=cfg.get("embed_dim", 256),  
            depth=cfg.get("depth", 4),            
            heads=cfg.get("heads", 8)             
        ).to(device)
    else:
        raise ValueError(f"Unknown architecture option: {cfg.model_name}")
        
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))
    
    wd = cfg.get("weight_decay", 1e-4)
    if cfg.model_name == "UltimateResNet":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=wd)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=wd)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    
    wandb.watch(model, log="all", log_freq=10)
    
    for epoch in range(cfg.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step() 
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        print(f"Epoch [{epoch+1}/{cfg.epochs}] -> Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    
    print("\n➔ Running Final Evaluation with Test-Time Augmentation (TTA)...")
    tta_loss, tta_acc = validate_with_tta(model, val_loader, criterion, device)
    
    print(f"🔥 FINAL STANDARDS -> Normal Val Acc: {val_acc:.4f} | TTA BOOSTED VAL ACC: {tta_acc:.4f} 🔥")
    wandb.log({"val/tta_accuracy": tta_acc, "val/tta_loss": tta_loss})    
    
    wandb.finish()