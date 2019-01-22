from torch.optim import Adam
from utils.metrics import PerformanceEntry
import torch
from tqdm import tqdm
from utils.constants import tasks_label_count

from net import BaseTagger
from net.taggers.general_tagger import GeneralTagger
from net.modules.descr import DescrModule, DescrBinaryModule, DescrRankedModule
import os
from sklearn.metrics import confusion_matrix


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

	def fit(self, dataloader, track_loss=None):

		self.model.train()

		losses = list()

		for i, (descr, labels) in enumerate(dataloader):

			descr = descr.to(device=self.device)
			labels = labels.to(device=self.device)

			preds = self.model(descr)

			loss = self.model.loss(preds, labels)

			losses.append(loss.item())

			self.optimizer.zero_grad()

			loss.backward()

			self.optimizer.step()

			# if i % 10 == 0 or (i+1) == len(dataloader):
			# 	print('Progress Fit: [{}/{}] Batch Loss: {}'.format(i+1, len(dataloader), loss.item()))

		self.optimizer.zero_grad()

		return sum(losses)/len(dataloader) if track_loss else None

	def predict(self, dataloader, info=None, eval_col=None):
		pass

	def evaluate(self, dataloader, info=None, eval_col='vnctr'):

		self.model.eval()

		performance = PerformanceEntry(num_classes=self.num_classes)

		y_true_list, y_score_list = list(), list()

		with torch.no_grad():

			for i, (descr, labels) in enumerate(dataloader):

				descr = descr.to(device=self.device)
				labels = labels.to(device=self.device)

				preds = self.model(descr)

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
