import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

_FMINST_CLASSES = {
  "T-shirt/top": 0,
  "Trouser": 1,
  "Pullover": 2,
  "Dress": 3,
  "Coat": 4,
  "Sandal": 5,
  "Shirt": 6,
  "Sneaker": 7,
  "Bag": 8
}

class AutoencoderFMNIST(nn.Module):
    def __init__(self, output_dim=5):
        super(AutoencoderFMNIST, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, output_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, 784), nn.Sigmoid()
        )
        self.classifier = nn.Linear(output_dim, 2)

    def forward(self, x):
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        logits = self.classifier(z)
        return reconstruction, logits, z

def filter_classes(dataset, class_A, class_B):
    indices = (dataset.targets == class_A) | (dataset.targets == class_B)
    X = dataset.data[indices].float().view(-1, 784) / 255.0
    y = (dataset.targets[indices] == class_B).long().numpy()
    return X, y

def load_fmnist(class_a, class_b, data_dir='./data'):
  class_a_idx = _FMINST_CLASSES[class_a]
  class_b_idx = _FMINST_CLASSES[class_b]
  transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
  fmnist_train = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
  fmnist_test = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
  X_train_full, y_train_full = filter_classes(fmnist_train, class_a_idx, class_b_idx)
  X_test_full, y_test_full = filter_classes(fmnist_test, class_a_idx, class_b_idx)
  return {
     "X_train": X_train_full,
     "y_train": y_train_full,
     "X_test": X_test_full,
     "y_test": y_test_full
  }

