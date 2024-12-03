"""
This model is inspired on the model proposed by the article: 
Kim,J.H.,Kim,B.G.,Roy,P.P.&Jeong,D.M.Efficient facial expression 
recognition algorithm based on hierarchical deep neural network 
structure. IEEE Access 7, 41273–41285 (2019).
"""

import torch
from torch import nn

class CNNImages(nn.Module):
  def __init__(self, input_shape : int, hidden_units : int, output_shape : int):
    super().__init__()
    self.block_one = nn.Sequential(
        nn.Conv2d(
            in_channels = input_shape,
            out_channels = hidden_units,
            kernel_size = 5,
            stride = 1,
            padding = 2
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2)
    )

    self.block_two = nn.Sequential(
        nn.Conv2d(
            in_channels = hidden_units,
            out_channels = 2 * hidden_units,
            kernel_size = 5,
            stride = 1,
            padding = 2
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2)
    )

    self.block_three = nn.Sequential(
        nn.Conv2d(
            in_channels = 2 * hidden_units,
            out_channels = 4 * hidden_units,
            kernel_size = 5,
            stride = 1,
            padding = 2
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2)
    )

    self.fc_1 = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features = hidden_units * 1024, out_features = 1024),
        nn.ReLU()
    )

    self.classifier = nn.Sequential(
        nn.Linear(in_features = 1024, out_features = output_shape)
    )

  def forward(self, x : torch.Tensor) -> torch.Tensor:
    x = self.block_one(x)
    x = self.block_two(x)
    x = self.block_three(x)
    x = self.fc_1(x)
    x = self.classifier(x)
    return x

if __name__ == "__main__": 
  print("CNN model definition!")