import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# 1. Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Image Preprocessing (matching training specs)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Load Model Architecture & Weights
weights_path = os.path.join("backend", "app", "ml", "weights", "quality_mobilenetv3_best.pth")
model = models.mobilenet_v3_large(weights=None)
num_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(num_features, 3)

model.load_state_dict(torch.load(weights_path, map_location=device))
model.to(device)
model.eval()

# 4. Class Labels (alphabetical order as detected by ImageFolder)
classes = ['fresh', 'poor', 'regular']

def predict_seaweed(image_path):
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    print(f"\nImage: {image_path}")
    print(f"Predicted Quality: {classes[predicted_idx]} ({confidence.item() * 100:.2f}% confidence)")

# Example test: Replace with an actual image path on your drive
test_image_path = "gusoB.jpg"
if os.path.exists(test_image_path):
    predict_seaweed(test_image_path)
else:
    print(f"File not found: {test_image_path}. Please update the path to a valid image.")