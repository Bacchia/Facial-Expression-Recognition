import torch
import wandb
from src.models import LinearBaselineModel

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return running_loss / total, correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
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

def run_experiment(config, train_loader, val_loader):
    wandb.init(project="FER2013-Project", name=config.get("experiment_name", "nameless_run"), config=config)
    cfg = wandb.config
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if cfg.model_name == "LinearBaseline":
        model = LinearBaselineModel().to(device)
    else:
        raise ValueError(f"Unknown model architecture: {cfg.model_name}")
        
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    
    wandb.watch(model, log="all", log_freq=10)
    
    for epoch in range(cfg.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc
        })
        
        print(f"Epoch [{epoch+1}/{cfg.epochs}] -> Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
    wandb.finish()