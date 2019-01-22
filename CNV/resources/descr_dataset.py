from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
from utils.constants import pandas_cols, smiles_alphabet
from resources.utils import transform_prediction_binary
from resources.utils import transform_prediction_classification
from resources.utils import transform_prediction_ranked
from resources.utils import chain_predictions_binary
from resources.utils import chain_predictions_classification
from resources.utils import labels_to_binary
from resources.utils import labels_to_ranked

from resources.base_dataset import BaseDataset

class DescrDataset(BaseDataset):

	def __init__(self, csv_file, descr_file, eval=None):
		super().__init__(csv_file, eval)

		self.descr_file = pd.read_csv(descr_file)

	def __getitem__(self, idx):

		inchikey = self.data_file['inchikey'].iloc[idx]

		# DESCR
		descr = self.descr_file[self.descr_file['inchikey'] == inchikey].iloc[0, 3:].values
		descr = torch.from_numpy(descr.astype(np.float32))

		# LABELS
		if self.eval:
			labels = self.data_file['DILI'].iloc[idx].astype(np.float)
		else:
			labels = self.data_file[pandas_cols].iloc[idx].values.astype(np.float)
		labels = torch.from_numpy(labels).to(dtype=torch.float)

		sample = {'descr': descr, 'labels': labels}

		return sample

	def transform_prediction(self, y_pred, y_true, eval_col='vnctr'):
		return transform_prediction_classification(y_pred=y_pred, y_true=y_true, eval_col=eval_col)

	def chain_predictions(self, preds):
		return chain_predictions_classification(preds)

	def collate_fn(self, data):
		descr, labels = list(), list()
		for item in data:
			descr.append(item['descr'])
			labels.append(item['labels'])

		# stack descr
		descr = torch.stack(descr, 0)

		# stack labels
		labels = torch.stack(labels, 0)

		return descr, labels


class DescrBinaryDS(DescrDataset):

	def __getitem__(self, idx):
		sample = super().__getitem__(idx)

		sample['labels'] = labels_to_binary(sample['labels'])

		return sample

	def transform_prediction(self, y_pred, y_true, eval_col='vnctr'):
		return transform_prediction_binary(y_pred=y_pred, y_true=y_true, eval_col=eval_col)

	def chain_predictions(self, preds):
		return chain_predictions_binary(preds)


class DescrRankedDS(DescrDataset):

	def __getitem__(self, idx):
		sample = super().__getitem__(idx)

		sample['labels'] = labels_to_ranked(sample['labels'])

		return sample

	def transform_prediction(self, y_pred, y_true, eval_col='vnctr'):
		return transform_prediction_ranked(y_pred=y_pred, y_true=y_true, eval_col=eval_col)

	def chain_predictions (self, preds):
		return chain_predictions_binary(preds)