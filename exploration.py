import matplotlib.pyplot as plt
import random
import os
from utils import get_filtered_dataset

# Load dataset
dataset = get_filtered_dataset(train=True)

# Helper to undo normalization
def denormalize(img):
    img = img * 0.5 + 0.5  # undo normalization (mean=0.5, std=0.5)
    return img.clamp(0, 1)

# Pick 5 random samples
indices = random.sample(range(len(dataset)), 5)

fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for ax, idx in zip(axes, indices):
    img, label = dataset[idx]
    img = denormalize(img)
    ax.imshow(img.permute(1, 2, 0))
    ax.set_title("Airplane" if label == 0 else "Bird")
    ax.axis("off")

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/exploration.png")
plt.show()
