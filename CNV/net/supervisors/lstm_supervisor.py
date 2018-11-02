from sklearn.model_selection import KFold, GroupKFold
from torch.utils.data import DataLoader
from resources.subset import Subset
import os
import shutil

from tensorboardX import SummaryWriter

from resources.smiles_dataset import SmilesDataset
from tqdm import tqdm

from utils.metrics import PerformanceEntry

from net.taggers.lstm_tagger import LSTMTagger

from net import BaseSupervisor, BaseTagger

import time

from resources.smiles_dataset import collate_fn

import numpy as np

import copy


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

		fold_losses = list()
		fold_losses_valid = list()
		fold_accs = list()



		for i, (loader_train, loader_valid) in enumerate(self.fold_gen(kf)):
			print('\nFold: [{}/{}]\n'.format(i+1,folds))

			epoch_losses = list()
			epoch_losses_valid = list()
			epoch_accs = list()

			self.not_better = 0
			self.best_model_loss = np.inf
			self.best_tagger = self.tagger

			for epoch in range(self.settings.run.epochs):

				epoch_loss = self.tagger.fit(loader_train, track_loss=True)

				print('Fold: {} Epoch {}: Loss Train: {}'.format(i+1, epoch+1, epoch_loss))
				epoch_losses.append(epoch_loss)

				self.performance_train.loss = epoch_loss

				loss_valid = self._evaluate(loader_train, loader_valid, global_step=(i+1)*epoch)

				# TODO early stop

				# early stopping
				if self.settings.run.early_stop and epoch > 0:
					if self._test_early_stop(loss_valid, epoch+1):
						break

			fold_losses.append(epoch_losses)

			# add acc on validation set after last epoch to acc-list
			self.accs_fold.append(self.performance_valid.acc)

		print('mean fold accuracy on validation set: {}'.format(sum(self.accs_fold)/folds))
		print('loss fold x epoch', fold_losses)

		print(fo)


	def _evaluate(self, loader_train, loader_valid, global_step):

		self.performance_valid = self.tagger.evaluate(loader_valid, info='valid')

		print('Loss Valid: {}'.format(self.performance_valid.loss))
		print('Acc Valid: {}'.format(self.performance_valid.acc))

		self.summary_writer.add_scalars(
			main_tag=r'loss', tag_scalar_dict={
				r'training': self.performance_train.loss,
				r'validation': self.performance_valid.loss},
			global_step=global_step)

		self.summary_writer.add_scalars(
			main_tag=r'accuracy', tag_scalar_dict={
				# r'training': self.performance_train.acc,
				r'validation': self.performance_valid.acc},
			global_step=global_step)

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

