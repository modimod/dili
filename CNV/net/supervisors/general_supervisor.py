from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader
from resources.subset import Subset
from resources.transforms import Normalize
from resources.descr_dataset import DescrDataset
from resources.descriptor_cellpainting_dataset import DescrCellpaintingDataset
import os
import shutil
import time
import torch

from tensorboardX import SummaryWriter
from utils.metrics import PerformanceEntry
from net import BaseSupervisor

import copy
import json

from utils.constants import tasks_label_count


class GeneralSupervisor(BaseSupervisor):

	def __init__(self, settings):

		self.settings = settings

		self.tagger = None
		self.dataset = None
		self.len = None
		self.checkpoint_dir = None
		self.summary_dir = None

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

			transform = None
			if isinstance(self.dataset, DescrDataset) or isinstance(self.dataset, DescrCellpaintingDataset):
				mean, std = self.dataset.get_mean_std(train_idx)
				transform = Normalize(mean, std)

			self.dataset.transform = transform

			# Subset
			subset_train = Subset(self.dataset, train_idx)
			subset_valid = Subset(self.dataset, val_idx)

			loader_train = DataLoader(dataset=subset_train, batch_size=self.settings.run.batch_size, shuffle=True, collate_fn=self.dataset.collate_fn)
			loader_valid = DataLoader(dataset=subset_valid, batch_size=self.settings.run.batch_size_eval, shuffle=False, collate_fn=self.dataset.collate_fn)

			yield loader_train, loader_valid

	def cross_validate(self, folds=3):
		kf = GroupKFold(n_splits=folds)

		fold_perfs = list()

		for i, (loader_train, loader_valid) in enumerate(self.fold_gen(kf)):
			print('\nFold: [{}/{}]\n'.format(i+1,folds))

			self.tagger.reset()

			_, best_perf = self.fit(loader_train,loader_valid, epochs=self.settings.run.epochs, fold=i)

			# save best perfs to performance dir
			self._save_performance(best_perf, 'fold{}_best'.format(i+1))

			# add best perfs of all folds
			fold_perfs.append(best_perf)


		return fold_perfs

	def fit(self, loader_train, loader_valid, epochs, fold=0):

		self.not_better = 0

		best_perf = PerformanceEntry(num_classes=self.num_classes)
		best_model = self.tagger

		for epoch in range(epochs):
			print('Fold: {} Epoch: {}'.format(fold + 1, epoch + 1))

			train_loss = self.tagger.fit(loader_train, track_loss=True)
			#self.summary_writer.add_scalar('fold{}/train_loss'.format(fold + 1), train_loss, epoch + 1)

			valid_perf = self._evaluate(loader_valid, epoch=epoch, fold=fold)
			self._save_performance(valid_perf, 'fold_{}_epoch_{}'.format(fold + 1, epoch + 1))

			self.summary_writer.add_scalars('fold{}/losses'.format(fold + 1), {'train': train_loss, 'valid': valid_perf.loss}, epoch + 1)
			self.summary_writer.add_scalar('fold{}/diff'.format(fold + 1), valid_perf.loss - train_loss, epoch + 1)

			# early stopping
			if self.settings.run.early_stop and epoch > 0:
				if self._test_early_stop(valid_perf, best_perf, fold=fold, epoch=epoch):
					break

			if valid_perf.better_than(best_perf):
				best_perf = copy.deepcopy(valid_perf)
				best_model = copy.deepcopy(self.tagger)

		return best_model, best_perf

	def train_plain(self, epochs):

		self.not_better = 0

		transform = None
		if isinstance(self.dataset, DescrDataset) or isinstance(self.dataset, DescrCellpaintingDataset):
			mean, std = self.dataset.get_mean_std()
			transform = Normalize(mean, std)

		self.dataset.transform = transform

		loader_train = DataLoader(dataset=self.dataset, batch_size=self.settings.run.batch_size, shuffle=True,
								  collate_fn=self.dataset.collate_fn)

		for epoch in range(epochs):
			print('Epoch: {}'.format(epoch + 1))

			train_loss = self.tagger.fit(loader_train, track_loss=True)
			self.summary_writer.add_scalar('plain/train_loss', train_loss, epoch + 1)

			self._save_model(text='epoch_{}'.format(epoch+1))

		self._save_model(text='last_epoch')
		return self.tagger

	def _evaluate(self, loader_valid, epoch, fold=0):

		if isinstance(loader_valid.dataset, Subset):
			loader_valid.dataset.dataset.validation()

		perf = self.tagger.evaluate(loader_valid, info='valid', eval_col=self.settings.data.eval_col)

		if isinstance(loader_valid.dataset, Subset):
			loader_valid.dataset.dataset.training()

		# self.summary_writer.add_scalar('fold{}/valid_loss'.format(fold + 1), perf.loss, epoch + 1)
		# self.summary_writer.add_scalar('fold{}/valid_acc'.format(fold + 1), perf.accuracy, epoch + 1)
		# self.summary_writer.add_scalar('fold{}/valid_precision'.format(fold + 1), perf.precision, epoch + 1)
		# self.summary_writer.add_scalar('fold{}/valid_recall'.format(fold + 1), perf.recall, epoch + 1)
		# self.summary_writer.add_scalar('fold{}/valid_specificity'.format(fold + 1), perf.specificity, epoch + 1)
		# self.summary_writer.add_scalar('fold{}/valid_f1score'.format(fold + 1), perf.f1score, epoch + 1)
		# self.summary_writer.add_scalar('fold{}/valid_balanced_accuracy'.format(fold + 1), perf.balanced_accuracy,epoch + 1)

		return perf

	def _evaluate_train(self, loader_train, epoch, fold=0):

		loader_train.dataset.dataset.validation()
		perf = self.tagger.evaluate(loader_train, info='train', eval_col=self.settings.data.eval_col)
		loader_train.dataset.dataset.training()

		#self.summary_writer.add_scalar('fold{}/train_loss'.format(fold + 1), perf.loss, epoch + 1)
		self.summary_writer.add_scalar('fold{}/train_acc'.format(fold + 1), perf.accuracy, epoch + 1)
		self.summary_writer.add_scalar('fold{}/train_precision'.format(fold + 1), perf.precision, epoch + 1)
		self.summary_writer.add_scalar('fold{}/train_recall'.format(fold + 1), perf.recall, epoch + 1)
		self.summary_writer.add_scalar('fold{}/train_specificity'.format(fold + 1), perf.specificity, epoch + 1)
		self.summary_writer.add_scalar('fold{}/train_f1score'.format(fold + 1), perf.f1score, epoch + 1)
		self.summary_writer.add_scalar('fold{}/train_balanced_accuracy'.format(fold + 1), perf.balanced_accuracy,epoch + 1)

		return perf

	def reset(self) -> None:
		"""
		Reset tagger.
		"""
# 		torch.manual_seed(seed=self.__settings.run.seed)

		curr_time = str(int(time.time()))
		print('log directory: {}\n\n'.format(curr_time))
		self.base_dir = os.path.join(self.settings.log.log_dir, curr_time)

		self.checkpoint_dir = os.path.join(self.base_dir, 'checkpoint')
		self.summary_dir = os.path.join(self.base_dir, 'summary')
		self.performance_dir = os.path.join(self.base_dir, 'performance')

		self._prepare_directory(directory=self.checkpoint_dir)
		self._prepare_directory(directory=self.summary_dir)
		self._prepare_directory(directory=self.performance_dir)

		self.summary_writer = SummaryWriter(log_dir=self.summary_dir)

		self.settings.to_json_file(self.base_dir)

		self.dataset = self._create_dataset()

		self._create_model()

		self.len = len(self.dataset)

		self.num_classes = tasks_label_count[self.settings.data.eval_col]

		self.best_model_perf = PerformanceEntry(num_classes=self.num_classes)
		self.performance_folds = PerformanceEntry(num_classes=self.num_classes)

		self.accs_fold = list()

		self.losses_train_fold = list()
		self.losses_valid_fold = list()
		self.acc_valid_fold = list()

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
		pass

	def _create_model(self):
		pass

	def _test_early_stop(self, valid_perf, best_perf, fold, epoch):
		if valid_perf.better_than(best_perf):
			print('\n\nNew Best Model after Epoch {}!\n\n'.format(epoch+1))
			self.not_better = 0
		else:
			self.not_better += 1
			print('No new best model for [{}/{}] epochs\n'.format(self.not_better, self.settings.run.early_stop))

			if self.not_better >= self.settings.run.early_stop:
				return True
		return False

	def _save_model(self, text='', best=False):
		file_name = 'best_model_{}.pt'.format(text) if best else text+'.pt'

		checkpoint_file = os.path.join(self.checkpoint_dir, file_name)

		with open(checkpoint_file, mode=r'wb') as f:
			torch.save(obj=self.tagger, f=f)

	def _save_performance(self, perf, file_name):
		with open(os.path.join(self.performance_dir, '{}.json'.format(file_name)), mode=r'w') as f:
			json.dump(perf.to_dictionary(), f, indent=4)
