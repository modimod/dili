from torch.optim import Adam
from utils.metrics import PerformanceEntry
import torch
from tqdm import tqdm
from utils.constants import tasks_label_count

from net import BaseTagger
from net.taggers.general_tagger import GeneralTagger
from net.modules.lstm_gapnet import LSTMGapnetModule, LSTMGapnetBinaryModule, LSTMGapnetRankedModule
import os
from sklearn.metrics import confusion_matrix

class SmilesCellpaintingTagger(GeneralTagger):

	def _init_model(self):

		if self.settings.data.label_format == 'binary':
			m = LSTMGapnetBinaryModule
		elif self.settings.data.label_format == 'ranked':
			m = LSTMGapnetRankedModule
		else:
			m = LSTMGapnetModule

		self.model = m(self.settings, feature_extract=self.settings.architecture.feature_extract)
		self.model = self.model.to(device=self.device)

	def _init_optimizer(self):

		params_to_update = self.model.parameters()
		print("Params to learn:")
		if self.settings.architecture.feature_extract:
			params_to_update = []
			for name, param in self.model.named_parameters():
				if param.requires_grad == True:
					params_to_update.append(param)
					print("\t", name)
		else:
			for name, param in self.model.named_parameters():
				if param.requires_grad == True:
					print("\t", name)

		self.optimizer = Adam(params_to_update, lr=self.settings.optimiser.learning_rate)