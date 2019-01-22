from torch.utils.data import DataLoader
from resources.subset import Subset

class SplitterLearner:

	def __init__(self, splitter, dataset, supervisor, settings):
		self.splitter = splitter
		self.dataset = dataset
		self.supervisor = supervisor
		self.settings = settings

	def fit(self):

		for train_idx, val_idx in self.splitter.split(range(len(self.dataset)), groups=self.dataset.clusters):
			# Subset
			subset_train = Subset(self.dataset, train_idx)
			subset_valid = Subset(self.dataset, val_idx)

			loader_train = DataLoader(dataset=subset_train, batch_size=self.settings.run.batch_size, shuffle=True, collate_fn=collate_fn)
			loader_valid = DataLoader(dataset=subset_valid, batch_size=self.settings.run.batch_size_eval, shuffle=False, collate_fn=collate_fn)

			self.supervisor.train(loader_train, loader_valid)

	