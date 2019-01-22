from net.modules.CNV import CellpaintingCNV
from net.modules.gapnet import GAPNet02
from torch.optim import Adam
from utils.metrics import PerformanceEntry
import torch
from tqdm import tqdm
from utils.constants import tasks


from torch.nn.functional import log_softmax, sigmoid

from net import BaseTagger

class CNVTagger(BaseTagger):

	def __init__(self, settings, loss_functions=None):
		self.loss_functions = loss_functions

		self.settings = settings

		self.device = r'cuda' if torch.cuda.is_available() and self.settings.run.cuda else r'cpu'

		self._init_model()
		self._init_optimizer()

	def _init_model(self):

		self.model = CellpaintingCNV(loss_functions=self.loss_functions,
									 test_mode=self.settings.run.test_mode)
		self.model = self.model.to(device=self.device)

	def _init_optimizer(self):

		self.optimizer = Adam(self.model.parameters())

	def fit(self, dataloader, track_loss=None):

		self.model.train()

		losses = list()

		dataloader = tqdm(dataloader, total=len(dataloader), desc=r'Progress fit', unit=r'batch', leave=False)
		for i, batch in enumerate(dataloader):
			x = batch['image']
			y = batch['labels']

			x, y = x.to(device=self.device), y.to(device=self.device)
			#print('batch: [{}/{}]'.format(i,len(dataloader)))

			preds = self.model(x)

			loss = self.model.loss(preds, y)

			losses.append(loss.item())

			self.optimizer.zero_grad()

			loss.backward()

			self.optimizer.step()

		self.optimizer.zero_grad()

		return sum(losses)/len(dataloader) if track_loss else None

	def predict(self, dataloader, eval=None, info=None):

		predictions = [list() for _ in range(len(self.tasks))]

		labels = list()
		losses = list()

		with torch.no_grad():

			desc = r'Progress predict'
			if info: desc += ' {}'.format(info)

			dataloader = tqdm(dataloader, total=len(dataloader), desc=desc, unit=r'batch', leave=False)
			for i, batch in enumerate(dataloader):

				x = batch['image']
				x = x.to(device=self.device)

				preds = self.model(x)

				# TODO softmax or simply argmax?

				for p, ps in zip(predictions, preds):
					p.extend(ps)

				if eval:
					y = batch['labels']
					y = y.to(device=self.device)
					labels.extend(y)

					loss = self.model.loss(preds, y, 'vnctr')
					losses.append(loss.item())

		return predictions, (torch.stack(labels).cpu().numpy(), sum(losses)/len(dataloader)) if eval else None

	def evaluate(self, dataloader, info=None):

		self.model.eval()

		predictions, (labels, loss) = self.predict(dataloader, eval=True, info=info)

		performance = PerformanceEntry()
		performance.calc_accuracy(predictions, labels)
		performance.loss = loss

		return performance

	def reset(self):
		self._init_model()
		self._init_optimizer()