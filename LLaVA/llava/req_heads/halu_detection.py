import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class HaluDetectionHead30(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim1=1024, hidden_dim2=512, num_classes=3, dropout_p=0.1):
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
    def __init__(self, input_dim=4096, hidden_dim1=1024, hidden_dim2=512, num_classes=3, dropout_p=0.1):
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
    
    
    @torch.no_grad()
    def predict(self, x, return_probs=True):
        """
        Inference-time prediction.

        Args:
            x (Tensor): Input tensor of shape [N, input_dim].
            return_probs (bool): Whether to return probabilities or just class labels.

        Returns:
            If return_probs=True:
                (pred_labels, probs) — where
                    pred_labels: tensor of predicted class indices [N]
                    probs: tensor of softmax probabilities [N, num_classes]
            Else:
                pred_labels — only the class indices [N]
        """
        self.eval()  # disable dropout, batchnorm, etc.
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        return (preds, probs) if return_probs else preds

