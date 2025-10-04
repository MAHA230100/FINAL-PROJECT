from __future__ import annotations

import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
	def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
		super().__init__()
		self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
		self.head = nn.Sequential(
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, 1),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: (batch, seq_len, input_size)
		out, _ = self.lstm(x)
		last = out[:, -1, :]
		return self.head(last)
