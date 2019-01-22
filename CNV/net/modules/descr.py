import torch.nn as nn
import torch
from torch.nn import Linear, BCEWithLogitsLoss
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

from net.modules.base_modules import FCModule, FCDynamicModule
from net.modules.MultiOuts import MultiOuts, MultiOutsBinary
from utils.constants import pandas_cols, tasks_rank, tasks_idx, smiles_alphabet, descr_dim, tasks


class DescrModule(nn.Module):
	def __init__(self, settings):
		super().__init__()

		self.settings = settings

		self.fc = FCDynamicModule(
			input_dim=descr_dim,
			hidden_dims=self.settings.architecture.fc_hidden_dims,
			dropout=self.settings.architecture.fc_dropout
		)

		# multiout
		self.multiout_in = self.fc.sizes[-1]
		self.multiout = MultiOuts(self.multiout_in)

		# loss
		self.loss = self.multiout.masked_loss

	def forward (self, descr):
		fc_out = self.fc(descr)

		return self.multiout(fc_out)


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


class DescrBinaryModule(DescrModule):
	def __init__(self, settings):
		super().__init__(settings)

		self.multiout = MultiOutsBinary(self.multiout_in, len(pandas_cols))
		self.loss = self.multiout.loss


class DescrRankedModule(DescrModule):
	def __init__(self, settings):
		super().__init__(settings)

		self.multiout = MultiOutsBinary(self.multiout_in, sum(tasks_rank.values()))
		self.loss = self.multiout.loss




