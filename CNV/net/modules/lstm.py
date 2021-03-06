import torch.nn as nn
import torch
from torch.nn import Linear, BCEWithLogitsLoss
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

from net.modules.base_modules import LSTMModule
from net.modules.MultiOuts import MultiOuts, MultiOutsBinary
from utils.constants import pandas_cols, tasks, tasks_rank, tasks_idx, smiles_alphabet


class SmilesModule(nn.Module):
	def __init__(self, settings):
		super().__init__()

		self.settings = settings

		self.lstm = LSTMModule(
			input_dim=self.settings.architecture.lstm_sliding_window * len(smiles_alphabet),
			hidden_dim=self.settings.architecture.lstm_hidden_dim,
			num_layers=self.settings.architecture.lstm_num_layers,
			dropout=self.settings.architecture.lstm_dropout)

		# multiout
		self.multiout_in = self.settings.architecture.lstm_hidden_dim
		self.multiout = MultiOuts(self.multiout_in)

		# loss
		self.loss = self.multiout.masked_loss

	def forward (self, smiles, lengths):
		lstm_out = self.lstm(smiles, lengths)

		return self.multiout(lstm_out)


def masked_loss_binary(outputs, targets, spec_target=None):
	loss = BCEWithLogitsLoss()

	if spec_target:
		idx = tasks_idx[spec_target]
		targets = targets[:, idx] if len(targets.shape) > 1 else targets
		outputs = outputs[:, idx]

	mask = (targets != -1).to(dtype=torch.float)
	outputs = outputs * mask
	targets = targets * mask

	return loss(outputs, targets)


class SmilesBinaryModule(SmilesModule):
	def __init__(self, settings):
		super().__init__(settings)

		self.multiout = MultiOutsBinary(self.multiout_in, len(tasks))
		self.loss = self.multiout.loss


class SmilesRankedModule(SmilesModule):
	def __init__(self, settings):
		super().__init__(settings)

		self.multiout = MultiOutsBinary(self.multiout_in, sum(tasks_rank.values()))
		self.loss = self.multiout.loss


