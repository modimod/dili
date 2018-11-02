from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import pandas as pd
import numpy as np
import torch
from utils.constants import pandas_cols, smiles_alphabet


class SmilesDataset(Dataset):

	def __init__(self, csv_file, sliding_window=1):
		self.features, self.labels, self.clusters = load_data(csv_file=csv_file)

		self.alphabet = smiles_alphabet
		self.one_hot = self._one_hot()

		self.sliding_window = sliding_window

	def __len__(self):
		return len(self.features)

	def __getitem__(self, i):
		features = self._prepare_input(self.features[i][0])

		labels = None
		if self.labels is not None:
			labels = self._prepare_targets(self.labels[i])
			return features, labels

		return features

	def _prepare_input(self,smiles_sequence):
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

def collate_fn(data):
	"""
	:param data: list of tuple (sequence, label).
	 	- sequence: torch tensor of shape (?, 56)
	 	- label: torch tensor of shape (1)

	:return:
	 	features: torch tensor of shape (batch_size, padded_length, one_hot_length (window_size*len(alphabet))
	 	labels: torch tensor of shape (batch_size, 1)
	 	lengths: list; valid length for each padded sequence
	"""

	sequences, labels = None, None
	if len(data[0]) == 2:
		data.sort(key=lambda x: len(x[0]), reverse=True)
		sequences, labels = zip(*data)
	else:
		data.sort(key=lambda x: len(x), reverse=True)
		sequences = data


	# merge sequences
	lengths = [len(s) for s in sequences]

	features = torch.zeros(len(sequences), max(lengths), sequences[0].size()[1])
	for i, s in enumerate(sequences):
		end = lengths[i]
		features[i, :end] = s[:end]


	# merge labels
	if labels is not None:
		labels = torch.stack(labels, 0)

		return features, labels, lengths

	return features, lengths

def load_data(csv_file):
	data = pd.read_csv(csv_file)

	features = data['SMILES'].values
	labels = data[pandas_cols]
	clusters = data['cluster'].values

	# remove rows that have no labels defined
	# -1 means unknown; return False if row consists of -1 only
	mask = labels.apply(axis=1, func=lambda row: not np.all(row == -1))

	labels = labels.values[mask, :]
	features = features[mask]
	clusters = clusters[mask]

	return features, labels, clusters
