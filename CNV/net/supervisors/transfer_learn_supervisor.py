from sklearn.model_selection import KFold
from torch.utils.data import DataLoader #Subset
from resources.subset import Subset
import os
import shutil

from tensorboardX import SummaryWriter

from resources.cellpainting_dataset import CellpaintingDataset
from tqdm import tqdm

from utils.metrics import PerformanceEntry

from net.modules.transfer_learn_models import get_data_transforms

from net.taggers.transfer_learning_tagger import TransferLearningTagger

from net import BaseSupervisor, BaseTagger

import numpy as np


class TransferLearningSupervisor(BaseSupervisor):

	def __init__(self, settings, model_name, feature_extract=True):

		self.settings = settings

		self.tagger = None
		self.dataset = None
		self.len = None

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
			subset_train.dataset.transform = self.transforms['train']

			subset_valid = Subset(self.dataset, val_idx)
			subset_train.dataset.transform = self.transforms['val']

			loader_train = DataLoader(dataset=subset_train, batch_size=self.settings.run.batch_size, shuffle=True)
			loader_valid = DataLoader(dataset=subset_valid, batch_size=self.settings.run.batch_size_eval, shuffle=False)

			yield loader_train, loader_valid

	def train(self, folds=3):
		'''
		:param folds:
		:return:
		'''

		kf = KFold(n_splits=folds)

		fold_losses = list()

		for i, (loader_train, loader_valid) in enumerate(self.fold_gen(kf)):
			print('Fold: [{}/{}]'.format(i+1,folds))

			epoch_losses = list()

			for epoch in tqdm(range(1, self.settings.run.epochs+1, 1),
							desc=r'Epochs', unit=r'ep', leave=False):

				epoch_loss = self.tagger.fit(loader_train, track_loss=True)
				print('Fold: {} Epoch {}: Loss Train: {}'.format(i+1, epoch+1, epoch_loss))
				epoch_losses.append(epoch_loss)

				self.performance_train.loss = epoch_loss

				self._evaluate(loader_train, loader_valid, global_step=(i+1)*epoch)

				# TODO early stop

			fold_losses.append(epoch_losses)

			# add acc on validation set after last epoch to acc-list
			self.accs_fold.append(self.performance_valid.acc)

		print('mean fold accuracy on validation set: {}'.format(sum(self.accs_fold)/folds))

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


	def reset (self) -> None:
		"""
		Reset tagger.
		"""
		#torch.manual_seed(seed=self.__settings.run.seed)

		self._prepare_directory(directory=self.settings.log.checkpoint_directory)
		self._prepare_directory(directory=self.settings.log.summary_directory)
		self.summary_writer = SummaryWriter(log_dir=self.settings.log.summary_directory)

		self.dataset = self._create_dataset()

		if self.tagger is not None and isinstance(self.tagger, BaseTagger):
			self.tagger.reset()
		else:
			self.tagger = TransferLearningTagger(self.settings, self.model_name, self.feature_extract)

		self.len = len(self.dataset)

		self.performance_train = PerformanceEntry()
		self.performance_valid = PerformanceEntry()

		self.accs_fold = list()

		self.transforms = get_data_transforms(self.tagger.input_size)

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





