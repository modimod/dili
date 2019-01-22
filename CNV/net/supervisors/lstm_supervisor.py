from sklearn.model_selection import KFold, GroupKFold
from torch.utils.data import DataLoader
from resources.subset import Subset
import os
import shutil

from tensorboardX import SummaryWriter

from resources.smiles_dataset import SmilesDataset
from tqdm import tqdm

from utils.metrics import PerformanceEntry

from net.taggers.lstm_tagger import SmilesTagger

from net import BaseSupervisor, BaseTagger

import time

import numpy as np

import copy

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau


class LSTMSupervisor(BaseSupervisor):

	def __init__(self, settings):

		self.settings = settings

		self.tagger = None
		self.dataset = None
		self.len = None

		self.summary_writer = None

		self.reset()

	def __del__ (self) -> None:
		"""
		Close summary writer instance.
		"""
		if self.summary_writer is not None and isinstance(self.summary_writer, SummaryWriter):
			self.summary_writer.close()

	def fold_gen(self, splitter):

		for train_idx, val_idx in splitter.split(range(self.len), groups=self.dataset.clusters):
			# Subset
			subset_train = Subset(self.dataset, train_idx)
			subset_valid = Subset(self.dataset, val_idx)

			loader_train = DataLoader(dataset=subset_train, batch_size=self.settings.run.batch_size, shuffle=True, collate_fn=collate_fn)
			loader_valid = DataLoader(dataset=subset_valid, batch_size=self.settings.run.batch_size_eval, shuffle=False, collate_fn=collate_fn)

			yield loader_train, loader_valid

	def train(self, folds=3):
		print('start train')
		print('folds: {}'.format(folds))
		print('max epochs: {}'.format(self.settings.run.epochs))
		print()

		#kf = KFold(n_splits=folds, shuffle=True)

		kf = GroupKFold(n_splits=folds)

		for i, (loader_train, loader_valid) in enumerate(self.fold_gen(kf)):
			print('\nFold: [{}/{}]\n'.format(i+1,folds))

			self.tagger.reset()
			self._init_scheduler()


			losses_train = list()
			losses_valid = list()
			learning_rates = list()

			self.not_better = 0
			self.best_model_loss = np.inf
			self.best_tagger = self.tagger

			for epoch in range(self.settings.run.epochs):
				self.scheduler.step()
				learning_rates.append(self.scheduler.get_lr())

				train_loss = self.tagger.fit(loader_train, track_loss=True)

				print('Fold: {} Epoch {}: Loss Train: {}'.format(i+1, epoch+1, train_loss))
				losses_train.append(train_loss)

				valid_loss = self._evaluate(loader_train, loader_valid, global_step=(i+1)*epoch)
				losses_valid.append(valid_loss)

				#self.scheduler.step(valid_loss)

				# early stopping
				if self.settings.run.early_stop and epoch > 0:
					if self._test_early_stop(valid_loss, epoch+1):
						break



			self.losses_train_fold.append(losses_train)
			self.losses_valid_fold.append(losses_valid)
			self.learning_rates_fold.append(learning_rates)

			# add acc on validation set after last epoch to acc-list
			self.acc_valid_fold.append(self.performance_valid.acc)

	def _evaluate(self, loader_train, loader_valid, global_step):

		self.performance_valid = self.tagger.evaluate(loader_valid, info='valid')

		print('\nLoss Valid: {}'.format(self.performance_valid.loss))
		print('Acc Valid: {}\n'.format(self.performance_valid.acc))

		# self.summary_writer.add_scalars(
		# 	main_tag=r'loss', tag_scalar_dict={
		# 		r'training': self.performance_train.loss,
		# 		r'validation': self.performance_valid.loss},
		# 	global_step=global_step)
		#
		# self.summary_writer.add_scalars(
		# 	main_tag=r'accuracy', tag_scalar_dict={
		# 		# r'training': self.performance_train.acc,
		# 		r'validation': self.performance_valid.acc},
		# 	global_step=global_step)

		return self.performance_valid.loss


	def reset (self) -> None:
		"""
		Reset tagger.
		"""
		#torch.manual_seed(seed=self.__settings.run.seed)

		t = str(int(round(time.time() * 1000)))

		self._prepare_directory(directory=os.path.join(self.settings.log.checkpoint_directory, t))
		self._prepare_directory(directory=os.path.join(self.settings.log.summary_directory, t))
		self.summary_writer = SummaryWriter(log_dir=os.path.join(self.settings.log.summary_directory, t))

		self.dataset = self._create_dataset()

		if self.tagger is not None and isinstance(self.tagger, BaseTagger):
			self.tagger.reset()
		else:
			self.tagger = LSTMTagger(self.settings)

		self.len = len(self.dataset)

		self.performance_train = PerformanceEntry()
		self.performance_valid = PerformanceEntry()

		self.accs_fold = list()

		self.losses_train_fold = list()
		self.losses_valid_fold = list()
		self.acc_valid_fold = list()
		self.learning_rates_fold = list()

	def _init_scheduler(self):
		self.scheduler = StepLR(self.tagger.optimizer, step_size=10, gamma=0.1)
		#self.scheduler = ReduceLROnPlateau(self.tagger.optimizer, 'min')

	def _prepare_directory(self, directory: str) -> None:
		"""
		Prepare dictionary to be used by the supervisor.

		:param directory: dictionary to prepare
		"""
		if os.path.exists(directory):
			if self.settings.log.overwrite:
				shutil.rmtree(path=directory)
			else:
				raise FileExistsError(r'Target "{}" already existing! Aborting ...'.format(directory))
		os.makedirs(directory)


	def _create_dataset(self):
		return SmilesDataset(csv_file=self.settings.data.smiles_csv_file,
							 sliding_window=self.settings.architecture.lstm_sliding_window)

	def _test_early_stop(self, loss_valid, epoch):
		if loss_valid < self.best_model_loss:
			self.best_model_loss = loss_valid
			self.best_tagger = copy.deepcopy(self.tagger)
			print('\n\nNew Best Model after Epoch {}!\n\n'.format(epoch))
			self.not_better = 0
		else:
			self.not_better += 1
			print('No new best model for [{}/{}] epochs\n'.format(self.not_better, self.settings.run.early_stop))

			if self.not_better >= self.settings.run.early_stop:
				self.tagger = copy.deepcopy(self.best_tagger)
				return True

		return False

