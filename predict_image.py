import torch
from torchvision import transforms
from PIL import Image
from models import SimpleCNN

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("results/best_model.pth", map_location=device))
model.eval()

# Define transformations (same as training)
transform = transforms.Compose([
    transforms.Resize((32, 32)),    # CIFAR-10 image size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load your image
img_path = img_path = r"C:\Users\niran\OneDrive\Desktop\airplane.jpg" # Give the path of the image to be predicted
image = Image.open(img_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

# Predict
output = model(image_tensor)
_, predicted = torch.max(output, 1)

# Map to labels
labels = {0: "Airplane", 1: "Bird"}
print(f"Predicted label: {labels[predicted.item()]}")
