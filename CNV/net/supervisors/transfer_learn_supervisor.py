from sklearn.model_selection import KFold
from torch.utils.data import DataLoader #Subset
from resources.subset import Subset
import os
import shutil
import time
import torch

from tensorboardX import SummaryWriter

from resources.cellpainting_dataset import CellpaintingDataset
from tqdm import tqdm

from utils.metrics import PerformanceEntry

from net.modules.transfer_learn_models import get_data_transforms

from net.taggers.transfer_learning_tagger import TransferLearningTagger

from net import BaseSupervisor, BaseTagger

import numpy as np

import copy


class TransferLearningSupervisor(BaseSupervisor):

	def __init__(self, settings, model_name='gapnet', feature_extract=True):

		self.settings = settings

		self.tagger = None
		self.dataset = None
		self.len = None
		self.checkpoint_dir = None
		self.summary_dir = None

		self.model_name = model_name

		self.feature_extract = feature_extract

		self.summary_writer = None

		self.reset()

	def __del__ (self) -> None:
		"""
		Close summary writer instance.
		"""
		if self.summary_writer is not None and isinstance(self.summary_writer, SummaryWriter):
			self.summary_writer.close()

	def fold_gen(self, splitter):

		for train_idx, val_idx in splitter.split(range(self.len)):
			# Subset
			subset_train = Subset(self.dataset, train_idx)
			subset_train.dataset.transform = self.transforms['train'] if self.model_name != 'gapnet' else None

			subset_valid = Subset(self.dataset, val_idx)
			subset_train.dataset.transform = self.transforms['val'] if self.model_name != 'gapnet' else None

			loader_train = DataLoader(dataset=subset_train, batch_size=self.settings.run.batch_size, shuffle=True)
			loader_valid = DataLoader(dataset=subset_valid, batch_size=self.settings.run.batch_size_eval, shuffle=False)

			yield loader_train, loader_valid

	def train(self, folds=3):
		kf = KFold(n_splits=folds, shuffle=True)

		for i, (loader_train, loader_valid) in enumerate(self.fold_gen(kf)):
			print('\nFold: [{}/{}]\n'.format(i+1,folds))

			self.tagger.reset()

			self.not_better, self.best_model_loss, self.best_tagger = 0, np.inf, self.tagger

			for epoch in range(self.settings.run.epochs):

				train_loss = self.tagger.fit(loader_train, track_loss=True)
				self.summary_writer.add_scalar('fold{}/train_loss'.format(i+1), train_loss, epoch+1)

				valid_loss, valid_acc = self._evaluate(loader_train, loader_valid, fold=i, epoch=epoch)
				self.summary_writer.add_scalar('fold{}/valid_loss'.format(i+1), valid_loss, epoch+1)
				self.summary_writer.add_scalar('fold{}/valid_acc'.format(i+1), valid_acc, epoch+1)

				print('Fold: {} Epoch: {} Loss Train: {} Loss Valid: {} Acc Valid: {}'.format(i + 1, epoch + 1, train_loss,
																					valid_loss, valid_acc))

				self._save_model(text='fold{}_epoch_{}'.format(i + 1, epoch + 1), best=False)

				# early stopping
				if self.settings.run.early_stop and epoch > 0:
					if self._test_early_stop(valid_loss, epoch+1):
						break



			# add acc on validation set after last epoch to acc-list
			self.acc_valid_fold.append(self.performance_valid.acc)

	def _evaluate(self, loader_train, loader_valid, fold, epoch):

		self.performance_valid = self.tagger.evaluate(loader_valid, info='valid', eval_col=self.settings.data.eval_col)

		return self.performance_valid.loss, self.performance_valid.acc

	def reset (self) -> None:
		"""
		Reset tagger.
		"""
# 		torch.manual_seed(seed=self.__settings.run.seed)

		curr_time = str(int(time.time()))
		print('log directory: {}\n\n'.format(curr_time))
		base_dir = os.path.join(self.settings.log.log_dir, curr_time)

		self.checkpoint_dir = os.path.join(base_dir, 'checkpoint')
		self.summary_dir = os.path.join(base_dir, 'summary')

		self._prepare_directory(directory=self.checkpoint_dir)
		self._prepare_directory(directory=self.summary_dir)
		self.summary_writer = SummaryWriter(log_dir=self.summary_dir)

		self.dataset = self._create_dataset()

		if self.tagger is not None and isinstance(self.tagger, BaseTagger):
			self.tagger.reset()
		else:
			self.tagger = TransferLearningTagger(self.settings, self.model_name, self.feature_extract)

		self.len = len(self.dataset)

		self.performance_train = PerformanceEntry()
		self.performance_valid = PerformanceEntry()

		self.accs_fold = list()

		self.losses_train_fold = list()
		self.losses_valid_fold = list()
		self.acc_valid_fold = list()
		self.learning_rates_fold = list()


		self.transforms = get_data_transforms(self.tagger.input_size) if self.model_name != 'gapnet' else None

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
		return CellpaintingDataset(
			csv_file=self.settings.data.csv_file,
			root_dir=self.settings.data.root_dir,
			file_ext=self.settings.data.file_ext,
			#transform=Compose([ToTensor()]),
			mode_test=self.settings.run.test_mode)


	def _test_early_stop(self, loss_valid, epoch):
		if loss_valid < self.best_model_loss:
			self.best_model_loss = loss_valid
			self.best_tagger = copy.deepcopy(self.tagger)

			self._save_model(best=True)

			print('\n\nNew Best Model after Epoch {}!\n\n'.format(epoch))
			self.not_better = 0
		else:
			self.not_better += 1
			print('No new best model for [{}/{}] epochs\n'.format(self.not_better, self.settings.run.early_stop))

			if self.not_better >= self.settings.run.early_stop:
				self.tagger = copy.deepcopy(self.best_tagger)
				return True

		return False

	def _save_model(self, text='', best=False):
		file_name = 'best_model.pt' if best else text+'.pt'

		checkpoint_file = os.path.join(self.checkpoint_dir, file_name)

		with open(checkpoint_file, mode=r'wb') as f:
			torch.save(obj=self.tagger, f=f)
