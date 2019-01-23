from torch.optim import Adam
from net.taggers.general_tagger import GeneralTagger
from net.modules.lstm import SmilesModule, SmilesBinaryModule, SmilesRankedModule


class SmilesTagger(GeneralTagger):

	def __init__(self, settings):
		super().__init__(settings)

	def _init_model(self):

		if self.settings.data.label_format == 'binary':
			m = SmilesBinaryModule
		elif self.settings.data.label_format == 'ranked':
			m = SmilesRankedModule
		else:
			m = SmilesModule

		self.model = m(self.settings)
		self.model = self.model.to(device=self.device)

	def _init_optimizer(self):

		self.optimizer = Adam(self.model.parameters(), lr=self.settings.optimiser.learning_rate)
