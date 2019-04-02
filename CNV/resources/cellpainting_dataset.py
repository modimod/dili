import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.constants import pandas_cols
from resources.utils import transform_prediction_binary
from resources.utils import transform_prediction_classification
from resources.utils import transform_prediction_ranked
from resources.utils import chain_predictions_binary
from resources.utils import chain_predictions_classification
from resources.utils import labels_to_binary
from resources.utils import labels_to_ranked
from resources.base_dataset import BaseDataset

class CellpaintingDataset(BaseDataset):

	"""Cellpainting dataset."""

	def __init__(self, csv_file, npzs_file, root_dir, file_ext, transform=None, mode_test=None, eval=None):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			npzs_file (string): Path to csv file that connects inchikey and sample-keys/images (npzs_inchi_reduced_dilirank.csv)
			root_dir (string): Directory with all the images.
			file_ext (string): File extension
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		super().__init__(csv_file, eval)

		self.npzs_file = pd.read_csv(npzs_file)
		self.len = len(self.npzs_file)
		self.clusters = np.arange(self.len)

		if mode_test:
			self.npzs_file['SAMPLE_KEY'] = '26247-P17-5'


		if mode_test:
			self.npzs_file = self.npzs_file.sample(100)
			self.clusters = np.arange(len(self.npzs_file))
			self.len = len(self.npzs_file)

		self.root_dir = root_dir
		self.file_ext = file_ext
		self.transform = transform

	def __getitem__(self, idx):
		row = self.npzs_file.iloc[idx]
		inchikey = row['INCHIKEY']
		sample_key = row['SAMPLE_KEY']

		# IMAGE
		img_name = os.path.join(self.root_dir, '{}.npz'.format(sample_key))
		image = np.load(img_name)
		image = image['sample'].astype(np.float32)
		image = image.transpose(2,0,1) #C,L,W
		image = torch.from_numpy(image)

		# LABELS
		if self.eval:
			labels = self.data_file.loc[self.data_file.inchikey == inchikey, ['DILI']].iloc[0].values.astype(np.float)
		else:
			labels = self.data_file.loc[self.data_file.inchikey == inchikey, pandas_cols].iloc[0].values.astype(np.float)

		labels = torch.from_numpy(labels).to(dtype=torch.float)
		sample = {'image': image, 'labels': labels}

		if self.transform:
			sample = self.transform(sample)

		return sample

	def transform_prediction(self, y_pred, y_true, eval_col='vnctr'):
		return transform_prediction_classification(y_pred=y_pred, y_true=y_true, eval_col=eval_col)

	def chain_predictions(self, preds):
		return chain_predictions_classification(preds)

	def collate_fn(self, data):

		images, labels = list(), list()
		for item in data:
			images.append(item['image'])
			labels.append(item['labels'])

		# stack images
		images = torch.stack(images, 0)

		# stack labels
		labels = torch.stack(labels, 0)

		return {'features': images, 'labels': labels}

class CellpaintingBinaryDS(CellpaintingDataset):

	def __getitem__(self, idx):
		sample = super().__getitem__(idx)

		sample['labels'] = labels_to_binary(sample['labels'])

		# TODO treat severity column different (maybe all above 4 is DILI positive)

		return sample

	def transform_prediction(self, y_pred, y_true, eval_col='vnctr'):
		return transform_prediction_binary(y_pred=y_pred, y_true=y_true, eval_col=eval_col)

	def chain_predictions(self, preds):
		return chain_predictions_binary(preds)


class CellpaintingRankedDS(CellpaintingDataset):

	def __getitem__(self, idx):
		sample = super().__getitem__(idx)

		sample['labels'] = labels_to_ranked(sample['labels'])

		return sample

	def transform_prediction(self, y_pred, y_true, eval_col='vnctr'):
		return transform_prediction_ranked(y_pred=y_pred, y_true=y_true, eval_col=eval_col)

	def chain_predictions (self, preds):
		return chain_predictions_binary(preds)
