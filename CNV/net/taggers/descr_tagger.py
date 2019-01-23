from torch.optim import Adam
from net.taggers.general_tagger import GeneralTagger
from net.modules.descr import DescrModule, DescrBinaryModule, DescrRankedModule


class DescrTagger(GeneralTagger):

	def _init_model(self):

		if self.settings.data.label_format == 'binary':
			m = DescrBinaryModule
		elif self.settings.data.label_format == 'ranked':
			m = DescrRankedModule
		else:
			m = DescrModule

		self.model = m(self.settings)
		self.model = self.model.to(device=self.device)

	def _init_optimizer(self):

		self.optimizer = Adam(self.model.parameters(), lr=self.settings.optimiser.learning_rate)