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
from resources.descriptor_cellpainting_dataset import DescrCellpaintingDataset

from resources.utils import TensorList

from torch.utils.data.dataloader import default_collate


class DescrCellpaintingMultiDataset(DescrCellpaintingDataset):

	def __getitem__(self, idx):
		inchikey = self.data_file['inchikey'].iloc[idx]

		# DESCR
		descr = self.descr_file[self.descr_file['inchikey'] == inchikey].iloc[0, 3:].values
		descr = torch.from_numpy(descr.astype(np.float32))

		# IMAGES
		sample_keys = self.npzs_file.loc[self.npzs_file['INCHIKEY'] == inchikey, ['SAMPLE_KEY']].values

		# if no image for compound use black one (zeros)

		images = list()
		if len(sample_keys) == 0:
			images.append(torch.from_numpy(np.zeros((5, 520, 696), dtype=np.float32)))
		else:
			for k in sample_keys:
				img_name = os.path.join(self.root_dir, '{}.npz'.format(k[0]))
				image = np.load(img_name)

				image = image['sample'].astype(np.float32)
				image = image.transpose(2, 0, 1) #C,L,W
				image = torch.from_numpy(image)

				images.append(image)

		images = torch.stack(images, 0)


		# LABELS
		if self.eval:
			labels = self.data_file['DILI'].iloc[idx].astype(np.float)
		else:
			labels = self.data_file[pandas_cols].iloc[idx].values.astype(np.float)
		labels = torch.from_numpy(labels).to(dtype=torch.float)

		sample = {'descr': descr, 'image': images, 'labels': labels}

		#if self.transform:
		#	sample = self.transform(sample)

		return sample

	def collate_fn(self, data):

		descr, labels = list(), list()
		images = TensorList()

		for item in data:
			descr.append(item['descr'])
			images.append(item['image'])
			labels.append(item['labels'])

		# stack descr
		descr = torch.stack(descr, 0)

		# stack images
		#images = torch.stack(images, 0)

		# stack labels
		labels = torch.stack(labels, 0)

		return {'features': [descr, images], 'labels': labels}


class DescrCellpaintingMultiBinaryDS(DescrCellpaintingMultiDataset):

	def __getitem__(self, idx):
		sample = super().__getitem__(idx)

		sample['labels'] = labels_to_binary(sample['labels'])

		# TODO treat severity column different (maybe all above 4 is DILI positive)

		return sample

	def transform_prediction(self, y_pred, y_true, eval_col='vnctr'):
		return transform_prediction_binary(y_pred=y_pred, y_true=y_true, eval_col=eval_col)

	def chain_predictions(self, preds):
		return chain_predictions_binary(preds)


class DescrCellpaintingMultiRankedDS(DescrCellpaintingMultiDataset):

	def __getitem__(self, idx):
		sample = super().__getitem__(idx)

		sample['labels'] = labels_to_ranked(sample['labels'])

		return sample

	def transform_prediction(self, y_pred, y_true, eval_col='vnctr'):
		return transform_prediction_ranked(y_pred=y_pred, y_true=y_true, eval_col=eval_col)

	def chain_predictions (self, preds):
		return chain_predictions_binary(preds)

