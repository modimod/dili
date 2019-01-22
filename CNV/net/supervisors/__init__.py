from resources.smiles_cellpainting_dataset import SmilesCellpaintingDataset, SmilesCellpaintingBinaryDS, SmilesCellpaintingRankedDS
from resources.descriptor_cellpainting_dataset import DescrCellpaintingDataset, DescrCellpaintingBinaryDS, DescrCellpaintingRankedDS
from resources.smiles_dataset import SmilesDataset, SmilesBinaryDS, SmilesRankedDS
from resources.descr_dataset import DescrDataset, DescrBinaryDS, DescrRankedDS
from resources.cellpainting_dataset import CellpaintingDataset, CellpaintingBinaryDS, CellpaintingRankedDS

from .general_supervisor import GeneralSupervisor

from net.taggers.smiles_cellpainting_tagger import SmilesCellpaintingTagger
from net.taggers.descr_cellpainting_tagger import DescrCellpaintingTagger
from net.taggers.lstm_tagger import SmilesTagger
from net.taggers.descr_tagger import DescrTagger
from net.taggers.gapnet_tagger import GapnetTagger
from net.taggers.general_tagger import GeneralTagger


class SmilesCellpaintingSupervisor(GeneralSupervisor):

	def _create_dataset(self):
		if self.settings.data.label_format == 'binary':
			ds = SmilesCellpaintingBinaryDS
		elif self.settings.data.label_format == 'ranked':
			ds = SmilesCellpaintingRankedDS
		else:
			ds = SmilesCellpaintingDataset

		return ds(
			csv_file=self.settings.data.csv_file,
			npzs_file=self.settings.data.npzs_file,
			root_dir=self.settings.data.root_dir,
			file_ext=self.settings.data.file_ext,
			mode_test=self.settings.run.test_mode,
			sliding_window=self.settings.architecture.lstm_sliding_window)

	def _create_model(self):
		if self.tagger is not None and isinstance(self.tagger, GeneralTagger):
			self.tagger.reset()
		else:
			self.tagger = SmilesCellpaintingTagger(settings=self.settings)


class DescrCellpaintingSupervisor(GeneralSupervisor):

	def _create_dataset(self):
		if self.settings.data.label_format == 'binary':
			ds = DescrCellpaintingBinaryDS
		elif self.settings.data.label_format == 'ranked':
			ds = DescrCellpaintingRankedDS
		else:
			ds = DescrCellpaintingDataset

		return ds(
			csv_file=self.settings.data.csv_file,
			npzs_file=self.settings.data.npzs_file,
			descr_file=self.settings.data.descr_file,
			root_dir=self.settings.data.root_dir,
			file_ext=self.settings.data.file_ext,
			mode_test=self.settings.run.test_mode)

	def _create_model (self):
		if self.tagger is not None and isinstance(self.tagger, GeneralTagger):
			self.tagger.reset()
		else:
			self.tagger = DescrCellpaintingTagger(settings=self.settings)


class SmilesSupervisor(GeneralSupervisor):

	def _create_dataset(self):
		if self.settings.data.label_format == 'binary':
			ds = SmilesBinaryDS
		elif self.settings.data.label_format == 'ranked':
			ds = SmilesRankedDS
		else:
			ds = SmilesDataset

		return ds(csv_file=self.settings.data.csv_file,
					sliding_window=self.settings.architecture.lstm_sliding_window)

	def _create_model(self):
		if self.tagger is not None and isinstance(self.tagger, GeneralTagger):
			self.tagger.reset()
		else:
			self.tagger = SmilesTagger(settings=self.settings)


class DescrSupervisor(GeneralSupervisor):

	def _create_dataset(self):
		if self.settings.data.label_format == 'binary':
			ds = DescrBinaryDS
		elif self.settings.data.label_format == 'ranked':
			ds = DescrRankedDS
		else:
			ds = DescrDataset

		return ds(
			csv_file=self.settings.data.csv_file,
			descr_file=self.settings.data.descr_file)

	def _create_model (self):
		if self.tagger is not None and isinstance(self.tagger, GeneralTagger):
			self.tagger.reset()
		else:
			self.tagger = DescrTagger(settings=self.settings)


class GapnetSupervisor(GeneralSupervisor):

	def _create_dataset(self):
		if self.settings.data.label_format == 'binary':
			ds = CellpaintingBinaryDS
		elif self.settings.data.label_format == 'ranked':
			ds = CellpaintingRankedDS
		else:
			ds = CellpaintingDataset

		return ds(
			csv_file=self.settings.data.csv_file,
			npzs_file=self.settings.data.npzs_file,
			root_dir=self.settings.data.root_dir,
			file_ext=self.settings.data.file_ext,
			mode_test=self.settings.run.test_mode)

	def _create_model (self):
		if self.tagger is not None and isinstance(self.tagger, GeneralTagger):
			self.tagger.reset()
		else:
			self.tagger = GapnetTagger(settings=self.settings)