import torch
import torch.nn as nn


import torch
import torch.nn as nn

class HaluDetectionHead30(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim1=8192, hidden_dim2=4096, num_classes=3, dropout_p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim2, num_classes)
        )

    def forward(self, x):
        return self.net(x)

    

import torch
import torch.nn as nn

class HaluDetectionHead24(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim1=8192, hidden_dim2=4096, num_classes=3, dropout_p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim2, num_classes)
        )

    def forward(self, x):
        return self.net(x)

    