import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

class FilteredCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root='./data', train=True, transform=None):
        super().__init__(root=root, train=train, download=True, transform=transform)
        target_classes = [0, 2]  # airplane=0, bird=2
        self.indices = [i for i, t in enumerate(self.targets) if t in target_classes]
        # Remap labels: airplane -> 0, bird -> 1
        self.targets = [0 if self.targets[i] == 0 else 1 for i in self.indices]
        self.data = self.data[self.indices]

def get_filtered_dataset(train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return FilteredCIFAR10(root='./data', train=train, transform=transform)

def plot_metrics(train_acc, val_acc, train_loss, val_loss):
    epochs = range(len(train_acc))
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, label='Train Accuracy')
    plt.plot(epochs, val_acc, label='Val Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.legend()
    plt.title('Loss')

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig('results/metrics.png')
    plt.close()
