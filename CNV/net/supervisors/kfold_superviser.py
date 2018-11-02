from sklearn.model_selection import KFold
from torch.utils.data import DataLoader #Subset
from resources.subset import Subset
import os
import shutil

from tensorboardX import SummaryWriter

from resources.cellpainting_dataset import CellpaintingDataset

from resources.transforms import ToTensor
from torchvision.transforms import Compose

from net.taggers.cnv_tagger import CNVTagger

from tqdm import tqdm

from utils.metrics import PerformanceEntry

from net import BaseSupervisor, BaseTagger

class KFoldSupervisor(BaseSupervisor):

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

	def train(self, folds=3):
		'''
		:param folds:
		:return:
		'''

		kf = KFold(n_splits=folds)

		for i, (train_idx, val_idx) in enumerate(tqdm(kf.split(range(self.len)), desc=r'Progress Fold', unit=r'f', leave=False)):
			# Subset
			subset_train = Subset(self.dataset, train_idx)
			subset_valid = Subset(self.dataset, val_idx)

			loader_train = DataLoader(dataset=subset_train, batch_size=self.settings.run.batch_size, shuffle=True)
			loader_valid = DataLoader(dataset=subset_valid, batch_size=self.settings.run.batch_size_eval, shuffle=False)

			for epoch in tqdm(range(1, self.settings.run.epochs+1, 1),
							desc=r'Epochs', unit=r'ep', leave=False):

				print('fold: {}, epoch: {}'.format(i,epoch))

				epoch_loss = self.tagger.fit(loader_train, track_loss=True)
				self.performance_train.loss = epoch_loss

				self._evaluate(loader_train, loader_valid, global_step=(i+1)*epoch)

			# add acc on validation set after last epoch to acc-list
			self.accs_fold.append(self.performance_valid.acc)

		print('mean fold accuracy on validation set: {}'.format(sum(self.accs_fold)/folds))

	def _evaluate(self, loader_train, loader_valid, global_step):

		#print('begin evaluate train')
		#performance_train = self.tagger.evaluate(loader_train, info='train')
		print('begin evalute valid')
		self.performance_valid = self.tagger.evaluate(loader_valid, info='valid')

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
		Reset sequence tagger.
		"""
		#torch.manual_seed(seed=self.__settings.run.seed)

		self._prepare_directory(directory=self.settings.log.checkpoint_directory)
		self._prepare_directory(directory=self.settings.log.summary_directory)
		self.summary_writer = SummaryWriter(log_dir=self.settings.log.summary_directory)

		self.dataset = self._create_dataset()

		if self.tagger is not None and isinstance(self.model, BaseTagger):
			self.tagger.reset()
		else:
			self.tagger = CNVTagger(settings=self.settings)

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
		return CellpaintingDataset(
			csv_file=self.settings.data.csv_file,
			root_dir=self.settings.data.root_dir,
			file_ext=self.settings.data.file_ext,
			transform=Compose([ToTensor()]),
			mode_test=self.settings.run.test_mode)





