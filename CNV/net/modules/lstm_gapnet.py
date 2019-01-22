import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torch.nn import BCEWithLogitsLoss

from net.modules.MultiOuts import MultiOuts, MultiOutsBinary
from net.modules.gapnet import GAPNet02
from net.modules.base_modules import LSTMModule
from utils.constants import smiles_alphabet, pandas_cols, tasks, tasks_rank, tasks_idx

class LSTMGapnetModule(nn.Module):
	def __init__ (self, settings, feature_extract=True):
		super().__init__()

		self.settings = settings
		self.feature_extract = feature_extract

		# lstm
		self.lstm = LSTMModule(input_dim=self.settings.architecture.lstm_sliding_window * len(smiles_alphabet),
								hidden_dim=self.settings.architecture.lstm_hidden_dim,
								num_layers=self.settings.architecture.lstm_num_layers,
								dropout=self.settings.architecture.lstm_dropout)

		# gapnet
		self._load_gapnet()


		# multiout
		self.multiout_in = self.settings.architecture.lstm_hidden_dim + self.num_ftrs
		self.multiout = MultiOuts(self.multiout_in)

		# loss
		self.loss = self.multiout.masked_loss

	def forward (self, smiles, lengths, images):
		lstm_out = self.lstm(smiles, lengths)
		gapnet_out = self.gapnet(images)

		x = torch.cat((lstm_out, gapnet_out), dim=1)

		return self.multiout(x)

	def _load_gapnet(self):
		self.gapnet = GAPNet02(input_shape=(5, 520, 696), fc_units=1024, dropout=0, num_classes=209)

		print('device: {}'.format(self.settings.run.device))
		checkpoint = torch.load(f=self.settings.data.pretrained_gapnet, map_location=self.settings.run.device)

		# rename checkpoint state_dict names
		new_state_dict = {k[7:]: v for k,v in checkpoint['state_dict'].items()}

		self.gapnet.load_state_dict(new_state_dict, strict=False)

		if self.feature_extract:
			for param in self.gapnet.parameters():
				param.requires_grad = False

		del self.gapnet.classifier[-1]  # remove last layer that was for classification

		self.num_ftrs = self.gapnet.classifier[3].out_features

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


class LSTMGapnetBinaryModule(LSTMGapnetModule):
	def __init__(self, settings, feature_extract=True):
		super().__init__(settings, feature_extract)

		self.multiout = MultiOutsBinary(self.multiout_in, len(tasks))
		self.loss = self.multiout.loss


class LSTMGapnetRankedModule(LSTMGapnetModule):
	def __init__(self, settings, feature_extract=True):
		super().__init__(settings, feature_extract)

		self.multiout = MultiOutsBinary(self.multiout_in, sum(tasks_rank.values()))
		self.loss = self.multiout.loss

