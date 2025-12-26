import torch
import torch.nn as nn
from torchvision import models,transforms

def get_helmet_model(WEIGHTS_PATH="weights/helmet_resnet18_best.pth", DEVICE="cuda" if torch.cuda.is_available() else "cpu"):
    """Load and return the helmet detection model."""
    helmet_model = models.resnet18(pretrained=False)
    helmet_model.fc = nn.Linear(helmet_model.fc.in_features, 2)
    helmet_model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    helmet_model.to(DEVICE)
    helmet_model.eval()
    return helmet_model

def get_helmet_model_for_training(DEVICE="cuda" if torch.cuda.is_available() else "cpu"):
    """Load and return the helmet detection model for training."""
    helmet_model = models.resnet18(pretrained=True)
    helmet_model.fc = nn.Linear(helmet_model.fc.in_features, 2)
    helmet_model.to(DEVICE)
    return helmet_model

def get_helmet_infer_transform(IMG_SIZE=224):
    """Return the transforms for helmet inference."""
    helmet_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])
    return helmet_transform

def get_helmet_train_transform(IMG_SIZE=224):
    """Return the transforms for helmet training."""
    helmet_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.3,0.3,0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                            std=[0.229,0.224,0.225])
    ])
    return helmet_transform

def get_helmet_val_transform(IMG_SIZE=224):
    """Return the transforms for helmet validation."""
    helmet_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                            std=[0.229,0.224,0.225])
    ])
    return helmet_transform