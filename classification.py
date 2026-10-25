import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# 1. Setup Device (Forces PyTorch to use your RTX 2050)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

# 2. Dataset Paths
BASE_DIR = os.path.join("dataset", "quality", "split")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "validation")

# 3. Data Augmentation
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Data Augmentaion for validation set
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # can be use for confusion matrix and classification report to get 20% random datasets
])
# This ensures that the validation set is not augmented, which is standard practice to evaluate the model's performance on unaltered data.
train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transforms)
val_dataset = datasets.ImageFolder(root=VAL_DIR, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# 4. Load MobileNetV3 and Modify the Classifier
print("Downloading pre-trained MobileNetV3 weights...")
weights = models.MobileNet_V3_Large_Weights.DEFAULT
model = models.mobilenet_v3_large(weights=weights)

# Replace the last layer to output 3 classes instead of 1000
num_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(num_features, 3)
model = model.to(device)

# 5. Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Training Loop
# we use 20 epochs to test the pipeline as of now.
EPOCHS = 20 # Running 30 epochs to test the pipeline
best_acc = 0.0

print("Starting training loop...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    # Validation Phase
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    val_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {running_loss/len(train_loader):.4f} | Validation Accuracy: {val_acc:.2f}%")
    
    # Save the model if it's the best one so far
    if val_acc >= best_acc:
        best_acc = val_acc
        # We save it into the same folder structure your groupmate uses
        os.makedirs(os.path.join("backend", "app", "ml", "weights"), exist_ok=True)
        torch.save(model.state_dict(), os.path.join("backend", "app", "ml", "weights", "quality_mobilenetv3_best.pth"))

print(f"Training complete! Best validation accuracy: {best_acc:.2f}%")
print("Weights saved to: backend/app/ml/weights/quality_mobilenetv3_best.pth")