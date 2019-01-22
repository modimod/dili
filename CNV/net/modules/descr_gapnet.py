import torch.nn as nn
import torch
from torch.nn import BCEWithLogitsLoss

from net.modules.MultiOuts import MultiOuts, MultiOutsBinary
from net.modules.gapnet import GAPNet02
from net.modules.base_modules import FCModule, FCDynamicModule
from utils.constants import pandas_cols, tasks, tasks_rank, tasks_idx, descr_dim

class DescrGapnetModule(nn.Module):
	def __init__ (self, settings, feature_extract=True):
		super().__init__()

		self.settings = settings
		self.feature_extract = feature_extract

		# descr
		self.descr = FCDynamicModule(
			input_dim=descr_dim,
			hidden_dims=self.settings.architecture.fc_hidden_dims,
			dropout=self.settings.architecture.fc_dropout
		)

		# gapnet
		self._load_gapnet()

		# multiout
		self.multiout_in = self.descr.sizes[-1] + self.num_ftrs
		self.multiout = MultiOuts(self.multiout_in)

		# loss
		self.loss = self.multiout.masked_loss

	def forward (self, descr, images):
		descr_out = self.descr(descr)
		gapnet_out = self.gapnet(images)

		x = torch.cat((descr_out, gapnet_out), dim=1)

		return self.multiout(x)

	def _load_gapnet(self):
		self.gapnet = GAPNet02(input_shape=(5, 520, 696), fc_units=1024, dropout=0, num_classes=209)

		print('device: {}'.format(self.settings.run.device))
		checkpoint = torch.load(f=self.settings.data.pretrained_gapnet, map_location=self.settings.run.device)

		# rename checkpoint state_dict names
		new_state_dict = {k[7:]: v for k,v in checkpoint['state_dict'].items()}

		self.gapnet.load_state_dict(new_state_dict)

		if self.feature_extract:
			for param in self.gapnet.parameters():
				param.requires_grad = False

		del self.gapnet.classifier[-1]  # remove last layer that was for classification

		self.num_ftrs = self.gapnet.classifier[3].out_features


class DescrGapnetBinaryModule(DescrGapnetModule):
	def __init__(self, settings, feature_extract=True):
		super().__init__(settings, feature_extract)

		self.multiout = MultiOutsBinary(self.multiout_in, len(tasks))
		self.loss = self.multiout.loss


class DescrGapnetRankedModule(DescrGapnetModule):
	def __init__(self, settings, feature_extract=True):
		super().__init__(settings, feature_extract)

		self.multiout = MultiOutsBinary(self.multiout_in, sum(tasks_rank.values()))
		self.loss = self.multiout.loss

