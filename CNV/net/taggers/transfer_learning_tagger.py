from torch.optim import Adam
from utils.metrics import PerformanceEntry
import torch
from tqdm import tqdm
from utils.constants import tasks

from net import BaseTagger
from net.modules.transfer_learn_models import initialize_model
import os

class TransferLearningTagger(BaseTagger):

	def __init__(self, settings, model_name='gapnet', feature_extract=True):

		self.settings = settings
		self.model_name = model_name
		self.feature_extract = feature_extract

		self.device = r'cuda' if torch.cuda.is_available() and self.settings.run.cuda else r'cpu'

		self._init_model()
		self._init_optimizer()

		if self.settings.run.checkpoint_file:
			self._restore_model()

	def _init_model(self):

		self.model, self.input_size = initialize_model(self.settings, model_name=self.model_name, num_classes=len(tasks),
									  feature_extract=self.feature_extract, use_pretrained=True)
		self.model = self.model.to(device=self.device)

	def _init_optimizer(self):

		params_to_update = self.model.parameters()
		print("Params to learn:")
		if self.feature_extract:
			params_to_update = []
			for name, param in self.model.named_parameters():
				if param.requires_grad == True:
					params_to_update.append(param)
					print("\t", name)
		else:
			for name, param in self.model.named_parameters():
				if param.requires_grad == True:
					print("\t", name)

		self.optimizer = Adam(params_to_update, lr=self.settings.optimiser.learning_rate)

	def fit(self, dataloader, track_loss=None):

		self.model.train()

		losses = list()

		for i, batch in enumerate(dataloader):

			x = batch['image']
			y = batch['labels']

			x, y = x.to(device=self.device), y.to(device=self.device)

			x = x.float()
			preds = self.model(x)

			loss = self.model.loss(preds, y)

			losses.append(loss.item())

			self.optimizer.zero_grad()

			loss.backward()

			self.optimizer.step()

			if i % 10 == 0 or (i+1) == len(dataloader):
				print('Progress Fit: [{}/{}] Batch Loss: {}'.format(i+1, len(dataloader), loss.item()))

		self.optimizer.zero_grad()

		return sum(losses)/len(dataloader) if track_loss else None

	def predict(self, dataloader, eval=None, info=None, eval_col=None):

		predictions = [list() for _ in range(len(tasks))]

		labels = list()
		losses = list()

		with torch.no_grad():

			for i, batch in enumerate(dataloader):

				x = batch['image']
				x = x.to(device=self.device)
				x = x.float()

				preds = self.model(x)

				for p, ps in zip(predictions, preds):
					p.extend(ps)

				if eval:
					y = batch['labels']
					y = y.to(device=self.device)
					labels.extend(y)

					#loss = self.model.loss(preds, y, 'vnctr')
					loss = self.model.loss(preds, y, eval_col)
					if i % 10 == 0 or (i + 1) == len(dataloader):
						print('Progress Predict {}: [{}/{}] Batch Loss: {}'.format(info, i + 1, len(dataloader), loss.item()))

					losses.append(loss.item())

		return predictions, (torch.stack(labels).cpu().numpy(), sum(losses)/len(dataloader)) if eval else None

	def evaluate(self, dataloader, info=None, eval_col='vnctr'):

		self.model.eval()

		predictions, (labels, loss) = self.predict(dataloader, eval=True, info=info, eval_col=eval_col)

		performance = PerformanceEntry()
		performance.calc_accuracy(predictions, labels)
		performance.loss = loss

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
			#start_epoch = checkpoint['epoch']
			self.model.load_state_dict(checkpoint['state_dict'])
			#self.optimizer.load_state_dict(checkpoint['optimizer'])
			print("=> loaded checkpoint '{}' (epoch {})"
				  .format(self.settings.run.checkpoint_file, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(self.settings.run.checkpoint_file))