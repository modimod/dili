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

from torch.utils.data.dataloader import default_collate


class DescrCellpaintingDataset(BaseDataset):

	"""Descriptor & Cellpainting dataset."""

	def __init__(self, csv_file, descr_file, npzs_file, root_dir, file_ext, transform=None, mode_test=None, eval=None):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations and cluster-entry (dilirank_wo_test_cluster.csv)
			descr_file (string): Path to csv file that connects PubChem_CID and Descriptor-Data	(descr_all.csv)
			npzs_file (string): Path to csv file that connects inchikey and sample-keys/images (npzs_inchi_reduced.csv)
			root_dir (string): Directory with all the images.
			file_ext (string): File extension
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		super().__init__(csv_file, eval)

		if mode_test:
			self.data_file = self.data_file[:100]
			self.clusters = self.data_file['cluster']
			self.len = len(self.data_file)


		self.descr_file = pd.read_csv(descr_file)
		self.npzs_file = pd.read_csv(npzs_file)

		if mode_test:
			self.npzs_file['SAMPLE_KEY'] = '26247-P17-5'

		self.root_dir = root_dir
		self.file_ext = file_ext
		self.transform = transform

	def __getitem__(self, idx):
		inchikey = self.data_file['inchikey'].iloc[idx]

		# DESCR
		descr = self.descr_file[self.descr_file['inchikey'] == inchikey].iloc[0, 3:].values
		descr = torch.from_numpy(descr.astype(np.float32))

		# IMAGE
		sample_keys = self.npzs_file[self.npzs_file['INCHIKEY'] == inchikey]

		# if training fold sample one image
		# else if validation fold take first image
		if self.validation_fold:
			sample_key = sample_keys['SAMPLE_KEY'].values[0] if len(sample_keys) > 0 else None
		else:
			sample_key = sample_keys.sample(n=1)['SAMPLE_KEY'].values[0] if len(sample_keys) > 0 else None

		# if no image for compound use black one (zeros)
		if sample_key is None:
			image = np.zeros((5, 520, 696), dtype=np.float32)
		else:
			img_name = os.path.join(self.root_dir, '{}.npz'.format(sample_key))
			image = np.load(img_name)

			image = image['sample'].astype(np.float32)
			image = image.transpose(2, 0, 1) #C,L,W
		image = torch.from_numpy(image)


		# LABELS
		if self.eval:
			labels = self.data_file['DILI'].iloc[idx].astype(np.float)
		else:
			labels = self.data_file[pandas_cols].iloc[idx].values.astype(np.float)
		labels = torch.from_numpy(labels).to(dtype=torch.float)

		sample = {'descr': descr, 'image': image, 'labels': labels}

		if self.transform:
			sample = self.transform(sample)

		return sample

	def transform_prediction(self, y_pred, y_true, eval_col='vnctr'):
		return transform_prediction_classification(y_pred=y_pred, y_true=y_true, eval_col=eval_col)

	def chain_predictions(self, preds):
		return chain_predictions_classification(preds)

	def collate_fn(self, data):

		descr, images, labels = list(), list(), list()
		for item in data:
			descr.append(item['descr'])
			images.append(item['image'])
			labels.append(item['labels'])

		# stack descr
		descr = torch.stack(descr, 0)

		# stack images
		images = torch.stack(images, 0)

		# stack labels
		labels = torch.stack(labels, 0)

		return descr, images, labels


class DescrCellpaintingBinaryDS(DescrCellpaintingDataset):

	def __getitem__(self, idx):
		sample = super().__getitem__(idx)

		sample['labels'] = labels_to_binary(sample['labels'])

		# TODO treat severity column different (maybe all above 4 is DILI positive)

		return sample

	def transform_prediction(self, y_pred, y_true, eval_col='vnctr'):
		return transform_prediction_binary(y_pred=y_pred, y_true=y_true, eval_col=eval_col)

	def chain_predictions(self, preds):
		return chain_predictions_binary(preds)


class DescrCellpaintingRankedDS(DescrCellpaintingDataset):

	def __getitem__(self, idx):
		sample = super().__getitem__(idx)

		sample['labels'] = labels_to_ranked(sample['labels'])

		return sample

	def transform_prediction(self, y_pred, y_true, eval_col='vnctr'):
		return transform_prediction_ranked(y_pred=y_pred, y_true=y_true, eval_col=eval_col)

	def chain_predictions (self, preds):
		return chain_predictions_binary(preds)

