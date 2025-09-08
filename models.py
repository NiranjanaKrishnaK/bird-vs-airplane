import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.LazyLinear(128),   # Automatically infers input features
            nn.ReLU(),
            nn.Linear(128, num_classes)  # Output: num_classes (default = 2)
        )

    def forward(self, x):
        return self.net(x)
