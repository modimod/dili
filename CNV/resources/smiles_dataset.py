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

class SmilesDataset(BaseDataset):

	def __init__(self, csv_file, sliding_window=3, eval=None):
		super().__init__(csv_file,eval)

		self.alphabet = smiles_alphabet
		self.one_hot = self._one_hot()

		self.sliding_window = sliding_window

	def __getitem__(self, idx):

		# SMILES
		smiles = self._prepare_input(self.data_file['SMILES'].iloc[idx])

		# LABELS
		if self.eval:
			labels = self.data_file[['DILI']].iloc[idx].values.astype(np.float)
		else:
			labels = self.data_file[pandas_cols].iloc[idx].values.astype(np.float)
		labels = torch.from_numpy(labels).to(dtype=torch.float)

		sample = {'smiles': smiles, 'labels': labels}

		return sample

	def _prepare_input(self, smiles_sequence):
		oh = np.array([self.one_hot[c] for c in smiles_sequence], dtype=np.int)

		# https://stackoverflow.com/questions/15722324/sliding-window-in-numpy
		def window_stack (a, stepsize=1, width=3):
			if len(a) == 2: # concat an empty array with 56
				return np.hstack([a[0],a[1],np.zeros(len(self.alphabet),dtype=int)])[None,:]
			if len(a) == 1:
				return np.hstack([a[0],np.zeros(len(self.alphabet),dtype=int),np.zeros(len(self.alphabet),dtype=int)])[None,:]
			return np.hstack(a[i:1 + i - width or None:stepsize] for i in range(0, width))

		oh = window_stack(oh, width=self.sliding_window)

		tensor = torch.from_numpy(oh).float()

		return tensor

	def _prepare_targets (self,targets):
		if np.isscalar(targets):
			targets = np.array([targets])
		tensor = torch.from_numpy(targets)

		return tensor.float()

	def _one_hot(self):
		one_hot = {}
		for i, l in enumerate(self.alphabet):
			bits = ['0'] * len(self.alphabet)
			bits[i] = '1'
			one_hot[l] = bits

		return one_hot

	def transform_prediction(self, y_pred, y_true, eval_col='vnctr'):
		return transform_prediction_classification(y_pred=y_pred, y_true=y_true, eval_col=eval_col)

	def chain_predictions(self, preds):
		return chain_predictions_classification(preds)

	def collate_fn(self, data):
		"""
		:param data: list of tuple (sequence, label).
		 	- sequence: torch tensor of shape (?, 56)
		 	- label: torch tensor of shape (1)

		:return:
		 	features: torch tensor of shape (batch_size, padded_length, one_hot_length (window_size*len(alphabet))
		 	labels: torch tensor of shape (batch_size, 1)
		 	lengths: list; valid length for each padded sequence
		"""

		newlist = sorted(data, key=lambda k: len(k['smiles']), reverse=True)

		smiles, labels, lengths = list(), list(), list()
		for item in newlist:
			smiles.append(item['smiles'])
			labels.append(item['labels'])
			lengths.append(len(item['smiles']))

		# pad smiles
		features = torch.zeros(len(data), max(lengths), smiles[0].size()[1])
		for i, s in enumerate(smiles):
			end = lengths[i]
			features[i, :end] = s[:end]

		# stack labels
		labels = torch.stack(labels, 0)

		return {'features': features, 'lengths': lengths, 'labels': labels}


class SmilesBinaryDS(SmilesDataset):

	def __getitem__(self, idx):
		sample = super().__getitem__(idx)

		sample['labels'] = labels_to_binary(sample['labels'], eval=self.eval)

		return sample

	def transform_prediction(self, y_pred, y_true, eval_col='vnctr'):
		return transform_prediction_binary(y_pred=y_pred, y_true=y_true, eval_col=eval_col)

	def chain_predictions(self, preds):
		return chain_predictions_binary(preds)


class SmilesRankedDS(SmilesDataset):

	def __getitem__(self, idx):
		sample = super().__getitem__(idx)

		sample['labels'] = labels_to_ranked(sample['labels'])

		return sample

	def transform_prediction(self, y_pred, y_true, eval_col='vnctr'):
		return transform_prediction_ranked(y_pred=y_pred, y_true=y_true, eval_col=eval_col)

	def chain_predictions (self, preds):
		return chain_predictions_binary(preds)