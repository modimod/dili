from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import pandas as pd
import numpy as np
import torch

class SmilesDataset(Dataset):

	def __init__(self, features, labels=None, sliding_window=1, alphabet = '#%()+-./0123456789=@ABCFGHIKLMNOPRSTVZ[\\]abcdeghilnorstu'):
		self.features = features

		self.labels = labels

		self.alphabet = alphabet
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

	sequences, labels = None,None
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

def load_data(features_file, labels_file=None):
	features = pd.read_csv(features_file).values

	# remove rows that have no labels defined
	if labels_file:
		labels = pd.read_csv(labels_file)

		mask = labels.apply(axis=1, func=lambda row: not np.all(row==-1)) # -1 means unknown; return False if row consists of -1 only

		labels = labels.values[mask,:]
		features = features[mask,:]

		return features,labels

	return features

def split_data(features,labels,split_perc=0.8):

	#shuffle dataset
	p = np.random.permutation(len(features))
	features = features[p]

	# split to train and test set
	x_train = features[:int(len(features) * split_perc)]
	x_test = features[int(len(features) * split_perc):]

	if labels is not None:
		labels = labels[p]
		y_train = labels[:int(len(features) * split_perc)]
		y_test = labels[int(len(features) * split_perc):]

		return x_train,y_train,x_test,y_test

	return x_train, x_test

def get_loader(features_file, labels_file=None,split_perc=0.8,batch_size=128,sliding_window=1):
	xx,yy = load_data(features_file=features_file, labels_file=labels_file)
	x_train, y_train, x_test, y_test = split_data(xx,yy,split_perc)

	smiles = SmilesDataset(features=x_train, labels=y_train, sliding_window=sliding_window)
	smiles_test = SmilesDataset(features=x_test, labels=y_test, sliding_window=sliding_window)

	dataloader = DataLoader(dataset=smiles, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
	dataloader_test = DataLoader(dataset=smiles_test, batch_size=len(smiles_test), collate_fn=collate_fn)

	return dataloader, dataloader_test