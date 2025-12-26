import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score
import numpy as np
import os
from utils.graph import plot_history
from models.helmet import get_helmet_model_for_training, get_helmet_train_transform, get_helmet_val_transform

# ------------------
# Config
# ------------------
ENABLE_CLASS_WEIGHTS = True
WEIGHTS_DIR = "weights/"
DATA_DIR = "data/helmet"   # make sure train/val subfolders exist
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------
# Transforms
# ------------------
train_transform = get_helmet_train_transform(IMG_SIZE=224)
val_transform = get_helmet_val_transform(IMG_SIZE=224)

# ------------------
# Datasets
# ------------------
print("Loading datasets...")
train_ds = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_transform)
val_ds   = datasets.ImageFolder(f"{DATA_DIR}/val", transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

# Check class mapping
print("Class to idx mapping:", train_ds.class_to_idx)
# Ensure 0=helmet, 1=no_helmet

# ------------------
# Compute class weights
# ------------------
class_counts = [0,0]
for _, label in train_ds:
    class_counts[label] += 1

total = sum(class_counts)
weights = [total / (2*c) for c in class_counts]  # formula for balanced weighting
class_weights = torch.tensor(weights).to(DEVICE)
print(f"Class weights: {class_weights}")

# ------------------
# Model
# ------------------
print("Setting up model...")
model = get_helmet_model_for_training(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights) if ENABLE_CLASS_WEIGHTS else nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_f1 = 0.0
history = {"loss": [], "val_loss": [],
           "f1": [], "val_f1": []}
# ------------------
# Training Loop
# ------------------
print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    all_preds = []
    all_labels = []
    total = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total+= labels.size(0)

    train_f1 = f1_score(all_labels, all_preds, average='macro')
    train_recall_nohelmet = recall_score(all_labels, all_preds, labels=[1], average='macro')

    # ------------------
    # Validation
    # ------------------
    model.eval()
    val_preds, val_labels = [], []
    val_loss = 0
    val_total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()*imgs.size(0)
            preds = outputs.argmax(dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
            val_total += labels.size(0)
        

    val_f1 = f1_score(val_labels, val_preds, average='macro')
    val_recall_nohelmet = recall_score(val_labels, val_preds, labels=[1], average='macro')

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {train_loss/len(train_loader):.4f} "
          f"Train F1: {train_f1:.3f} "
          f"Train Recall(NoHelmet): {train_recall_nohelmet:.3f} "
          f"Val F1: {val_f1:.3f} "
          f"Val Recall(NoHelmet): {val_recall_nohelmet:.3f}")
    
    # Record history
    history["loss"].append(train_loss/max(1,total))
    history["val_loss"].append(val_loss/max(1,val_total))
    history["f1"].append(train_f1)
    history["val_f1"].append(val_f1)

    # Save best model based on validation F1
    if val_f1 > best_f1:
        best_f1 = val_f1
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        torch.save(model.state_dict(), f"{WEIGHTS_DIR}/helmet_resnet18_best.pth")
        print("Saved new best model: helmet_resnet18_best.pth")

# ------------------
# Plot training history
plt = plot_history(history)
plt.show()
