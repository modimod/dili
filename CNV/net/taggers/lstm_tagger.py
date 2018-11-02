from net.modules.lstm import LSTMModule
from torch.optim import Adam
from utils.metrics import PerformanceEntry
import torch
from tqdm import tqdm
from utils.constants import smiles_alphabet, tasks

from net import BaseTagger

class LSTMTagger(BaseTagger):

	def __init__(self, settings):
		self.settings = settings

		self.device = r'cuda' if torch.cuda.is_available() and self.settings.run.cuda else r'cpu'

		self._init_model()
		self._init_optimizer()

	def _init_model(self):

		self.model = LSTMModule(input_dim=self.settings.architecture.lstm_sliding_window * len(smiles_alphabet),
								hidden_dim=self.settings.architecture.lstm_hidden_dim,
								num_layers=self.settings.architecture.lstm_num_layers,
								dropout=self.settings.architecture.lstm_dropout)
		self.model = self.model.to(device=self.device)

	def _init_optimizer(self):

		self.optimizer = Adam(self.model.parameters())

	def fit(self, dataloader, track_loss=None):

		self.model.train()

		losses = list()

		for i, (x,y,length) in enumerate(dataloader):

			self.optimizer.zero_grad()

			x, y = x.to(device=self.device), y.to(device=self.device)

			preds = self.model(x, length)

			loss = self.model.loss(preds, y)

			losses.append(loss.item())

			loss.backward()

			self.optimizer.step()

			if i % 10 == 0 or (i+1) == len(dataloader):
				print('Progress Fit: [{}/{}]'.format(i+1, len(dataloader)))

		self.optimizer.zero_grad()

		return sum(losses)/len(dataloader) if track_loss else None

	def predict(self, dataloader, eval=None, info=None):

		predictions = [list() for _ in range(len(tasks))]

		labels = list()
		losses = list()

		with torch.no_grad():
			for i, (x,y,length) in enumerate(dataloader):

				x = x.to(device=self.device)

				preds = self.model(x, length)

				for p, ps in zip(predictions, preds):
					p.extend(ps)

				if eval:
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
