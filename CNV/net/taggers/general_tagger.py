from utils.metrics import PerformanceEntry
import torch
from torch import nn
from utils.constants import tasks_label_count

from net import BaseTagger
import os
from sklearn.metrics import confusion_matrix
from resources.subset import Subset

class GeneralTagger(BaseTagger):

	def __init__(self, settings):

		self.settings = settings
		self.model = None

		self.device = settings.run.device

		if self.settings.data.label_format == 'binary':
			self.num_classes = 2
		else:
			self.num_classes = tasks_label_count[self.settings.data.eval_col]

		self.reset()

	def _init_model(self):
		pass

	def _model_to_device(self):
		if torch.cuda.device_count() > 1:
			print("Let's use", torch.cuda.device_count(), "GPUs!")
			self.model = nn.DataParallel(self.model)

		self.model = self.model.to(device=self.device)

	def _init_optimizer(self):
		pass

	def fit(self, dataloader, track_loss=None):

		self.model.train()

		losses = list()

		for i, data_dict in enumerate(dataloader):

			features = data_dict['features']
			labels = data_dict['labels']
			lengths = data_dict.get('lengths', None)

			if not isinstance(features, list):
				features = [features]

			for j in range(len(features)):
				features[j] = features[j].to(device=self.device)

			labels = labels.to(device=self.device)

			if lengths:
				preds = self.model(*features, lengths)
			else:
				preds = self.model(*features)

			if isinstance(self.model, nn.DataParallel):
				loss = self.model.module.loss(preds, labels)
			else:
				loss = self.model.loss(preds, labels)

			losses.append(loss.item())

			self.optimizer.zero_grad()

			loss.backward()

			self.optimizer.step()

			if i % 10 == 0 or (i+1) == len(dataloader):
				print('Progress Fit: [{}/{}]'.format(i+1, len(dataloader)))

		return sum(losses)/len(losses) if track_loss else None

	def predict(self, dataloader, info=None, eval_col=None):
		pass

	def evaluate(self, dataloader, info=None, eval_col='vnctr'):

		self.model.eval()

		performance = PerformanceEntry(num_classes=self.num_classes)

		y_true_list, y_score_list = list(), list()

		with torch.no_grad():

			for i, data_dict in enumerate(dataloader):

				features = data_dict['features']
				labels = data_dict['labels']
				lengths = data_dict.get('lengths', None)

				if not isinstance(features, list):
					features = [features]

				for j in range(len(features)):
					features[j] = features[j].to(device=self.device)

				labels = labels.to(device=self.device)

				if lengths:
					preds = self.model(*features, lengths)
				else:
					preds = self.model(*features)

				if isinstance(dataloader.dataset, Subset):
					y_pred, y_true, y_score = dataloader.dataset.dataset.transform_prediction(
						y_pred=preds,
						y_true=labels,
						eval_col=eval_col)
				else:
					y_pred, y_true, y_score = dataloader.dataset.transform_prediction(
						y_pred=preds,
						y_true=labels,
						eval_col=eval_col)

				if len(y_true) > 0:
					cm = confusion_matrix(y_pred=y_pred.cpu(), y_true=y_true.cpu(), labels=list(range(self.num_classes)))
					performance.update_confusion_matrix(cm)

					y_true_list.append(y_true)
					y_score_list.append(y_score)

				if isinstance(self.model, nn.DataParallel):
					loss = self.model.module.loss(preds, labels, eval_col)
				else:
					loss = self.model.loss(preds, labels, eval_col)

				performance.update_loss(loss.item())

		if self.settings.data.label_format == 'binary':
			y_true_all = torch.cat(y_true_list, dim=0)
			y_score_all = torch.cat(y_score_list, dim=0)

			performance.set_auc_scores(y_true=y_true_all.cpu(), y_score=y_score_all.cpu())

		return performance


	def reset(self):
		self._init_model()
		self._init_optimizer()
		if self.settings.run.checkpoint_file:
			self._restore_model()

	def _restore_model (self):
		if os.path.isfile(self.settings.run.checkpoint_file):
			print("=> loading checkpoint '{}'".format(self.settings.run.checkpoint_file))
			checkpoint = torch.load(self.settings.run.checkpoint_file)
			# start_epoch = checkpoint['epoch']
			self.model.load_state_dict(checkpoint['state_dict'])
			# self.optimizer.load_state_dict(checkpoint['optimizer'])
			print("=> loaded checkpoint '{}' (epoch {})"
				  .format(self.settings.run.checkpoint_file, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(self.settings.run.checkpoint_file))