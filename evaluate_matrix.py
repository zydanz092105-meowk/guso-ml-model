import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running evaluation on device: {device}")

# 2. Validation transforms matching your training script
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Exact dataset path from classification.py
VAL_DIR = os.path.join("dataset", "quality", "split", "validation")
print(f"Loading evaluation images from: {VAL_DIR}")

val_dataset = datasets.ImageFolder(root=VAL_DIR, transform=val_transforms)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
classes = val_dataset.classes
print(f"Detected classes ({len(classes)}): {classes}")

# 4. Rebuild MobileNetV3 Large architecture
model = models.mobilenet_v3_large(weights=None)
num_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(num_features, len(classes))

# 5. Load your saved weights
weights_path = os.path.join("backend", "app", "ml", "weights", "quality_mobilenetv3_best.pth")
if not os.path.exists(weights_path):
    print(f"Error: Weights file not found at {weights_path}")
    sys.exit(1)

model.load_state_dict(torch.load(weights_path, map_location=device))
model.to(device)
model.eval()

# 6. Run evaluation
all_preds = []
all_labels = []

print("Evaluating images with MobileNetV3 Large...")
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# 7. Print Classification Report
print("\n" + "=" * 45)
print("          CLASSIFICATION REPORT")
print("=" * 45)
print(classification_report(all_labels, all_preds, target_names=classes))

# 8. Generate and save Confusion Matrix plot
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    xticklabels=classes, 
    yticklabels=classes,
    cbar=False,
    annot_kws={"size": 14}
)
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.title('Guso Quality Classification - Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()

output_path = "confusion_matrix.png"
plt.savefig(output_path, dpi=300)
print(f"\n[+] Confusion matrix successfully saved to: {output_path}")

plt.show()