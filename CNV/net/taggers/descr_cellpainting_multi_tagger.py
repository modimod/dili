from torch.optim import Adam, SGD
from utils.metrics import PerformanceEntry
import torch
from tqdm import tqdm
from utils.constants import tasks_label_count

from net import BaseTagger
from net.taggers.general_tagger import GeneralTagger
from net.modules.descr_gapnet_multi import DescrGapnetMultiModule, DescrGapnetMultiBinaryModule, DescrGapnetMultiRankedModule
import os
from sklearn.metrics import confusion_matrix

from utils.multiple_optimizer import MultipleOptimizer


class DescrCellpaintingMultiTagger(GeneralTagger):

	def _init_model(self):

		if self.settings.data.label_format == 'binary':
			m = DescrGapnetMultiBinaryModule
		elif self.settings.data.label_format == 'ranked':
			m = DescrGapnetMultiRankedModule
		else:
			m = DescrGapnetMultiModule

		self.model = m(self.settings, feature_extract=self.settings.architecture.feature_extract)
		self._model_to_device()

	def _init_optimizer(self):
		if self.settings.architecture.feature_extract:
			params_to_update = []
			for name, param in self.model.named_parameters():
				if param.requires_grad == True:
					params_to_update.append(param)

			self.optimizer = Adam(params_to_update, lr=self.settings.optimiser.learning_rate)
		else:
			descr_opt = Adam(
				[{'params': self.model.descr.parameters()}, {'params': self.model.multiout.parameters()}]
				, lr=self.settings.optimiser.learning_rate)
			gapnet_opt = SGD(self.model.gapnet.parameters(), lr=self.settings.optimiser.learning_rate_gapnet)
			self.optimizer = MultipleOptimizer(descr_opt, gapnet_opt)
