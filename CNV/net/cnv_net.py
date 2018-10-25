from modules.CNV import CellpaintingCNV
from torch.optim import Adam
from utils.metrics import PerformanceEntry
import torch

class CNVNet():

	def __init__(self, tasks, loss_functions=None):
		self.tasks = tasks
		self.loss_functions = loss_functions

		self.device = r'cuda' if torch.cuda.is_available() and self.settings.run.cuda else r'cpu'

		self._init_model()
		self._init_optimizer()

	def _init_model(self):

		self.model = CellpaintingCNV(tasks=self.tasks, loss_functions=self.loss_functions)

	def _init_optimizer(self):

		self.optimizer = Adam(self.model.parameters())

	def fit(self, dataloader):

		self.model.train()

		for i, batch in enumerate(dataloader):
			x = batch['image']
			y = batch['labels']
			print('batch: [{}/{}]'.format(i,len(dataloader)))

			preds = self.model(x)

			loss = self.model.loss(preds, y)

			self.optimizer.zero_grad()

			loss.backward()

			self.optimizer.step()

		self.optimizer.zero_grad()

	def predict(self, data):

		self.model.eval()

		predictions = list()
		for i, (x,y) in enumerate(data):

			preds = self.model(x)
			# TODO softmax?

			predictions.extend(preds)

		return predictions


	def eval(self, data):

		self.model.eval()

		predictions = self.predict(data)

		performance = PerformanceEntry()
		performance.calc_accuracy(predictions, data.labels)

		return performance

	def reset(self):
		self._init_model()
		self._init_optimizer()