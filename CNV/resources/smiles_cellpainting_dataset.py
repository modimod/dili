import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.constants import pandas_cols, smiles_alphabet
from resources.utils import transform_prediction_binary
from resources.utils import transform_prediction_classification
from resources.utils import transform_prediction_ranked
from resources.utils import chain_predictions_binary
from resources.utils import chain_predictions_classification
from resources.utils import labels_to_binary
from resources.utils import labels_to_ranked

from resources.base_dataset import BaseDataset

class SmilesCellpaintingDataset(BaseDataset):

	"""SMILES & Cellpainting dataset."""

	def __init__(self, csv_file, npzs_file, root_dir, file_ext, transform=None, mode_test=None, eval=None, sliding_window=3):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			npzs_file (string):
			root_dir (string): Directory with all the images.
			file_ext (string): File extension
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		super().__init__(csv_file, eval)

		if mode_test:
			self.data_file = self.data_file.sample(100)
			self.clusters = self.data_file['cluster']
			self.len = len(self.data_file)

		self.npzs_file = pd.read_csv(npzs_file)

		self.alphabet = smiles_alphabet
		self.one_hot = self._one_hot()

		self.sliding_window = sliding_window

		# for testing
		if mode_test:
			self.npzs_file['SAMPLE_KEY'] = '26247-P17-5'

		self.root_dir = root_dir
		self.file_ext = file_ext
		self.transform = transform

	def __getitem__(self, idx):
		# SMILES
		smiles = self._prepare_input(self.data_file['SMILES'].iloc[idx])

		# IMAGE
		inchikey = self.data_file['inchikey'].iloc[idx]
		sample_keys = self.npzs_file[self.npzs_file['INCHIKEY'] == inchikey]

		# if training fold sample one image
		# else if validation fold take first image
		if self.validation_fold:
			sample_key = sample_keys['SAMPLE_KEY'].values[0] if len(sample_keys) > 0 else None
		else:
			sample_key = sample_keys.sample(n=1)['SAMPLE_KEY'].values[0] if len(sample_keys) > 0 else None

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

		sample = {'smiles': smiles, 'image': image, 'labels': labels}

		if self.transform:
			sample = self.transform(sample)

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

	def _prepare_targets (self, targets):
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
		:param data: list of dict entries {'smiles': smiles, 'image': image, 'labels': labels}.
			- smiles: torch tensor of shape (?, 56)
			- image: torch tensor of shape (5, 520, 696)
			- label: torch tensor of shape (7)

		:return:
			features: torch tensor of shape (batch_size, padded_length, one_hot_length (window_size*len(alphabet))
			images: torch tensor of shape (batch_size, 5, 520, 696)
			labels: torch tensor of shape (batch_size, 1)
			lengths: list; valid length for each padded sequence
		"""

		# sort descending by smiles len
		newlist = sorted(data, key=lambda k: len(k['smiles']), reverse=True)

		smiles, images, labels, lengths = list(), list(), list(), list()
		for item in newlist:
			smiles.append(item['smiles'])
			images.append(item['image'])
			labels.append(item['labels'])
			lengths.append(len(item['smiles']))

		# pad smiles
		features = torch.zeros(len(data), max(lengths), smiles[0].size()[1])
		for i, s in enumerate(smiles):
			end = lengths[i]
			features[i, :end] = s[:end]

		# stack images
		images = torch.stack(images, 0)

		# stack labels
		labels = torch.stack(labels, 0)

		return {'features': [features, images], 'lengths': lengths, 'labels': labels}


class SmilesCellpaintingBinaryDS(SmilesCellpaintingDataset):

	def __getitem__(self, idx):
		sample = super().__getitem__(idx)

		sample['labels'] = labels_to_binary(sample['labels'])

		# TODO treat severity column different (maybe all above 4 is DILI positive)

		return sample

	def transform_prediction(self, y_pred, y_true, eval_col='vnctr'):
		return transform_prediction_binary(y_pred=y_pred, y_true=y_true, eval_col=eval_col)

	def chain_predictions(self, preds):
		return chain_predictions_binary(preds)


class SmilesCellpaintingRankedDS(SmilesCellpaintingDataset):

	def __getitem__(self, idx):
		sample = super().__getitem__(idx)

		sample['labels'] = labels_to_ranked(sample['labels'])

		return sample

	def transform_prediction(self, y_pred, y_true, eval_col='vnctr'):
		return transform_prediction_ranked(y_pred=y_pred, y_true=y_true, eval_col=eval_col)

	def chain_predictions (self, preds):
		return chain_predictions_binary(preds)

