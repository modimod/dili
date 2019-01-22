from torch.optim import Adam
from utils.metrics import PerformanceEntry
import torch
from tqdm import tqdm
from utils.constants import tasks_label_count

from net import BaseTagger
from net.modules.lstm_gapnet import LSTMGapnetModule, LSTMGapnetBinaryModule, LSTMGapnetRankedModule
import os
from sklearn.metrics import confusion_matrix

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

	def _init_optimizer(self):
		pass

	def fit(self, dataloader, track_loss=None):
		pass

	def predict(self, dataloader, info=None, eval_col=None):
		pass

	def evaluate(self, dataloader, info=None, eval_col='vnctr'):
		pass

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