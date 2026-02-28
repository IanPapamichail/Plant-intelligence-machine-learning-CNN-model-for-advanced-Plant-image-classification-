import torch, os
import torch.nn as nn 
import torch.optim as optim 
from torchvision import datasets, transforms, models 
from torch.utils.data import DataLoader 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root="data/processed/triage/train", transform=transform)

val_dataset = datasets.ImageFolder(root="data/processed/triage/val", transform=transform)

