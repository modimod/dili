from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
import os
import shutil

from tensorboardX import SummaryWriter

from resources.cellpainting_dataset import CellpaintingDataset

from resources.transforms import ToTensor
from torchvision.transforms import Compose

from utils.constants import tasks
from net.cnv_net import CNVNet

from tqdm.auto import tqdm


class KFoldSupervisor:

	def __init__(self, settings):

		self.settings = settings

		self.model = None
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

		:param model: CNVNET (inkl optimizer)
		:param dataset: torch.utils.data.Dataset
		:param epochs:
		:param folds:
		:return:
		'''

		kf = KFold(n_splits=folds)

		for i,(train_idx, val_idx) in enumerate(kf.split(range(self.len))):
			# Subset
			subset_train = Subset(self.dataset, train_idx)
			subset_valid = Subset(self.dataset, val_idx)

			loader_train = DataLoader(dataset=subset_train, batch_size=self.settings.run.batch_size, shuffle=True)
			loader_valid = DataLoader(dataset=subset_valid, batch_size=self.settings.run.batch_size, shuffle=False)

			self._evaluate(loader_train, loader_valid, global_step=1)

			for epoch in tqdm(range(1, self.settings.run.epochs+1, 1),
							desc=r'Progress Fold [{}/{}]'.format(i, folds), unit=r'ep'):

				print('cv_epoch: {}, epoch: {}'.format(i,epoch))

				self.model.fit(loader_train)

				self._evaluate(loader_train, loader_valid, global_step=epoch)

	def _evaluate(self, loader_train, loader_valid, global_step):

		performance_train = self.model.eval(loader_train)
		performance_valid = self.model.eval(loader_valid)

		self.summary_writer.add_scalars(
			main_tag=r'loss', tag_scalar_dict={
				r'training': performance_train.loss,
				r'validation': performance_valid.loss},
			global_step=global_step)

		self.summary_writer.add_scalars(
			main_tag=r'accuracy', tag_scalar_dict={
				r'training': performance_train.accuracy,
				r'validation': performance_valid.accuracy},
			global_step=global_step)

		print('train: {}'.format(str(performance_train)))
		print('valid: {}'.format(str(performance_valid)))

	def reset (self) -> None:
		"""
		Reset sequence tagger.
		"""
		#torch.manual_seed(seed=self.__settings.run.seed)

		self._prepare_directory(directory=self.settings.log.checkpoint_directory)
		self._prepare_directory(directory=self.settings.log.summary_directory)
		self.summary_writer = SummaryWriter(log_dir=self.settings.log.summary_directory)

		self.dataset = self._create_dataset()

		if self.model is not None: # and isinstance(self.model, BaseTagger):
			self.model.reset()
		else:
			self.model = CNVNet(tasks=tasks, settings=self.settings)

		self.len = len(self.dataset)

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
			transform=Compose([ToTensor()]))





