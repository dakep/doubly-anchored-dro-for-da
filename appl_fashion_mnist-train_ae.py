import argparse

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from fashion_mnist_helper import load_fmnist, AutoencoderFMNIST

def _positive_int(value: str) -> int:
  parsed = int(value)
  if parsed <= 0:
    raise argparse.ArgumentTypeError("argument must be a positive integer")
  return parsed


def train_autoencoder(output_dim, class_a, class_b, data_dir='./data', epochs=100, lr=1e-3):
  print(f"Loading Fashion MNIST and filtering {class_a} vs {class_b}...")
  fmnist_data = load_fmnist(class_a, class_b, data_dir)

  print(f"Training Autoencoder on ALL {len(fmnist_data['X_train'])} Source Data images...")
  ae = AutoencoderFMNIST(output_dim=output_dim)
  criterion_mse = nn.MSELoss()
  criterion_ce = nn.CrossEntropyLoss()
  optimizer = optim.Adam(ae.parameters(), lr=lr)
  y_train_tensor = torch.tensor(fmnist_data['y_train'], dtype=torch.long)
  ae.train()
  for epoch in range(epochs):
      optimizer.zero_grad()
      reconstruction, logits, _ = ae(fmnist_data['X_train'])
      loss_mse = criterion_mse(reconstruction, fmnist_data['X_train'])
      loss_ce = criterion_ce(logits, y_train_tensor)
      loss = loss_mse + loss_ce
      loss.backward()
      optimizer.step()
      print(f"Epoch [{epoch + 1}/{epochs}] | MSE Loss: {loss.item():.4f}")

  return ae

def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("class_A", type=str)
  parser.add_argument("class_B", type=str)
  parser.add_argument("output_dim", type=_positive_int)
  parser.add_argument("encoder_output", type=str)
  args = parser.parse_args()
  ae = train_autoencoder(args.output_dim, args.class_A, args.class_B)
  torch.save(ae.state_dict(), args.encoder_output)
  print(f"Saved trained Autoencoder to {args.encoder_output}")

if __name__ == "__main__":
  main()
