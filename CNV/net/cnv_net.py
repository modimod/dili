from net.modules.CNV import CellpaintingCNV
from torch.optim import Adam
from utils.metrics import PerformanceEntry
import torch
from tqdm.auto import tqdm

class CNVNet():

	def __init__(self, settings, tasks, loss_functions=None):
		self.tasks = tasks
		self.loss_functions = loss_functions

		self.settings = settings

		self.device = r'cuda' if torch.cuda.is_available() and self.settings.run.cuda else r'cpu'

		self._init_model()
		self._init_optimizer()

	def _init_model(self):

		self.model = CellpaintingCNV(tasks=self.tasks, loss_functions=self.loss_functions)
		self.model = self.model.to(device=self.device)

	def _init_optimizer(self):

		self.optimizer = Adam(self.model.parameters())

	def fit(self, dataloader):

		self.model.train()

		dataloader = tqdm(dataloader, total=len(dataloader), desc=r'Progress fit', unit=r'batch', leave=False)
		for i, batch in enumerate(dataloader):
			x = batch['image']
			y = batch['labels']

			x, y = x.to(device=self.device), y.to(device=self.device)
			#print('batch: [{}/{}]'.format(i,len(dataloader)))

			preds = self.model(x)

			loss = self.model.loss(preds, y)

			self.optimizer.zero_grad()

			loss.backward()

			self.optimizer.step()

		self.optimizer.zero_grad()

	def predict(self, dataloader, eval=None):

		self.model.eval()

		predictions = list()
		labels = list()
		losses = list()

		with torch.no_grad():
			for i, batch in enumerate(dataloader):
				x = batch['image']
				x = x.to(device=self.device)

				preds = self.model(x)
				# TODO softmax?

				predictions.append(preds)

				if eval:
					y = batch['labels']
					y = y.to(device=self.device)
					labels.append(y)

					loss = self.model.loss(preds, y, 'vnctr')
					losses.append(loss.item)

		return predictions, (labels, sum(losses)/len(dataloader)) if eval else None


	def eval(self, dataloader):

		self.model.eval()

		predictions, (labels, loss) = self.predict(dataloader, eval=True)

		performance = PerformanceEntry()
		performance.calc_accuracy(predictions, labels)
		performance.update_loss(loss)

		return performance

	def reset(self):
		self._init_model()
		self._init_optimizer()