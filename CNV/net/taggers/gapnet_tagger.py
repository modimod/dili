from torch.optim import Adam, SGD
from net.taggers.general_tagger import GeneralTagger
from net.modules.gapnet_module import GapnetModule, GapnetBinaryModule, GapnetRankedModule
from utils.multiple_optimizer import MultipleOptimizer
from torch import nn

class GapnetTagger(GeneralTagger):

	def _init_model(self):

		if self.settings.data.label_format == 'binary':
			m = GapnetBinaryModule
		elif self.settings.data.label_format == 'ranked':
			m = GapnetRankedModule
		else:
			m = GapnetModule

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
			if isinstance(self.model, nn.DataParallel):
				multiout_opt = Adam(self.model.module.multiout.parameters(), lr=self.settings.optimiser.learning_rate)
				gapnet_opt = SGD(self.model.module.gapnet.parameters(), lr=self.settings.optimiser.learning_rate_gapnet)
			else:
				multiout_opt = Adam(self.model.multiout.parameters(), lr=self.settings.optimiser.learning_rate)
				gapnet_opt = SGD(self.model.gapnet.parameters(), lr=self.settings.optimiser.learning_rate_gapnet)
			self.optimizer = MultipleOptimizer(multiout_opt, gapnet_opt)
