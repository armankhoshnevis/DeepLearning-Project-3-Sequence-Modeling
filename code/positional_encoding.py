import torch  # type: ignore
import torch.nn as nn  # type: ignore
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5_000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pe = pe.to(device)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x
