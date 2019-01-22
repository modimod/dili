from torch.optim import Adam, SGD
from utils.metrics import PerformanceEntry
import torch
from tqdm import tqdm
from utils.constants import tasks_label_count

from net import BaseTagger
from net.taggers.general_tagger import GeneralTagger
from net.modules.gapnet_module import GapnetModule, GapnetBinaryModule, GapnetRankedModule
import os
from sklearn.metrics import confusion_matrix

from utils.multiple_optimizer import MultipleOptimizer


class GapnetTagger(GeneralTagger):

	def _init_model(self):

		if self.settings.data.label_format == 'binary':
			m = GapnetBinaryModule
		elif self.settings.data.label_format == 'ranked':
			m = GapnetRankedModule
		else:
			m = GapnetModule

		self.model = m(self.settings, feature_extract=self.settings.architecture.feature_extract)
		self.model = self.model.to(device=self.device)

	def _init_optimizer(self):
		if self.settings.architecture.feature_extract:
			params_to_update = []
			for name, param in self.model.named_parameters():
				if param.requires_grad == True:
					params_to_update.append(param)

			self.optimizer = Adam(params_to_update, lr=self.settings.optimiser.learning_rate)
		else:
			multiout_opt = Adam(self.model.multiout.parameters(), lr=self.settings.optimiser.learning_rate)
			gapnet_opt = SGD(self.model.gapnet.parameters(), lr=self.settings.optimiser.learning_rate_gapnet)
			self.optimizer = MultipleOptimizer(multiout_opt, gapnet_opt)


	def fit(self, dataloader, track_loss=None):

		self.model.train()

		losses = list()

		for i, (images, labels) in enumerate(dataloader):

			images = images.to(device=self.device)
			labels = labels.to(device=self.device)

			images = images.float()

			preds = self.model(images)

			loss = self.model.loss(preds, labels)

			losses.append(loss.item())

			self.optimizer.zero_grad()

			loss.backward()

			self.optimizer.step()

			if i % 10 == 0 or (i+1) == len(dataloader):
				print('Progress Fit: [{}/{}] Batch Loss: {}'.format(i+1, len(dataloader), loss.item()))

		self.optimizer.zero_grad()

		return sum(losses)/len(dataloader) if track_loss else None

	def predict(self, dataloader, info=None, eval_col=None):
		predictions = list()
		labels_list = list()
		losses = list()

		performance = PerformanceEntry(num_classes=tasks_label_count[eval_col])

		with torch.no_grad():

			for i, (smiles, images, labels, lengths) in enumerate(dataloader):

				smiles = smiles.to(device=self.device)
				images = images.to(device=self.device)
				labels = labels.to(device=self.device)

				images = images.float()

				preds = self.model(smiles, lengths, images)

				predictions.append(preds)

				labels_list.extend(labels)

				loss = self.model.loss(preds, labels, eval_col)

				losses.append(loss.item())

				if i % 10 == 0 or (i + 1) == len(dataloader):
					print('Progress Predict {}: [{}/{}] Batch Loss: {}'.format(info, i + 1, len(dataloader), loss.item()))

			predictions = dataloader.dataset.chain_predictions(predictions)

		return predictions, torch.stack(labels_list).cpu().numpy(), sum(losses)/len(dataloader)

	def evaluate(self, dataloader, info=None, eval_col='vnctr'):

		self.model.eval()

		performance = PerformanceEntry(num_classes=self.num_classes)

		y_true_list, y_score_list = list(), list()

		with torch.no_grad():

			for i, (images, labels) in enumerate(dataloader):

				images = images.to(device=self.device)
				labels = labels.to(device=self.device)

				images = images.float()

				preds = self.model(images)

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
