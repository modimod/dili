from unittest import TestCase
from resources.smiles_cellpainting_dataset import SmilesCellpaintingDataset
from torch import Tensor
import random

class TestSmilesCellpaintingDataset(TestCase):
	def setUp(self):
		ds = SmilesCellpaintingDataset(
			csv_file='FINAL_dilirank_wo_test_cluster.csv',
			root_dir='',
			file_ext='',
			mode_test=True,
			sliding_window=3)

		self.seed = 42

	def test__prepare_input (self):
		random.seed(self.seed)

		self.fail()

	def test__prepare_targets (self):
		self.fail()

	def test__one_hot (self):
		self.fail()

	def test_transform_prediction (self):
		self.fail()

	def test_chain_predictions (self):
		self.fail()
