from torch.optim import Adam

from utils.metrics import PerformanceEntry
import torch

from utils.constants import tasks, tasks_label_count

from sklearn.metrics import confusion_matrix
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

	def fit(self, dataloader, track_loss=None):

		self.model.train()

		losses = list()

		for i, (smiles, labels, lengths) in enumerate(dataloader):

			smiles = smiles.to(device=self.device)
			labels = labels.to(device=self.device)

			preds = self.model(smiles, lengths)

			loss = self.model.loss(preds, labels)

			losses.append(loss.item())

			self.optimizer.zero_grad()

			loss.backward()

			self.optimizer.step()

			if i % 10 == 0 or (i+1) == len(dataloader):
				print('Progress Fit: [{}/{}]'.format(i+1, len(dataloader)))

		self.optimizer.zero_grad()

		return sum(losses)/len(losses) if track_loss else None

	def predict(self, dataloader, info=None, eval_col=None):

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

	def evaluate(self, dataloader, info=None, eval_col='vnctr'):

		self.model.eval()

		performance = PerformanceEntry(num_classes=self.num_classes)

		y_true_list, y_score_list = list(), list()

		with torch.no_grad():

			for i, (smiles, labels, lengths) in enumerate(dataloader):

				smiles = smiles.to(device=self.device)
				labels = labels.to(device=self.device)

				preds = self.model(smiles, lengths)

				y_pred, y_true, y_score = dataloader.dataset.dataset.transform_prediction(
					y_pred=preds,
					y_true=labels,
					eval_col=eval_col)

				if len(y_true) > 0:
					cm = confusion_matrix(y_pred=y_pred, y_true=y_true, labels=list(range(self.num_classes)))
					performance.update_confusion_matrix(cm)

					y_true_list.append(y_true)
					y_score_list.append(y_score)

				loss = self.model.loss(preds, labels, eval_col)
				performance.update_loss(loss.item())

		if self.settings.data.label_format == 'binary':
			y_true_all = torch.cat(y_true_list, dim=0)
			y_score_all = torch.cat(y_score_list, dim=0)

			performance.set_auc_scores(y_true=y_true_all, y_score=y_score_all)

		return performance
