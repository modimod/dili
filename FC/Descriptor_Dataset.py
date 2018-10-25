from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch

class DescriptorDataset(Dataset):

	def __init__(self, features, labels=None):
		self.features = features

		self.labels = labels


	def __len__(self):
		return len(self.features)

	def __getitem__(self, i):
		features = self._prepare_input(self.features[i])

		labels = None
		if self.labels is not None:
			labels = self._prepare_targets(self.labels[i])
			return features, labels

		return features

	def _prepare_input(self,features):
		features = torch.from_numpy(features.astype(np.float)).float()
		#features = torch.nn.functional.normalize(features)
		return features

	def _prepare_targets (self,targets):

		if np.isscalar(targets):
			targets = np.array([targets])
		tensor = torch.from_numpy(targets)

		return tensor.float()


def load_data(features_file, labels_file=None):
	features = pd.read_csv(features_file).values
	features = features[:,1:]

	# remove rows that have no labels defined
	if labels_file:
		labels = pd.read_csv(labels_file)

		mask = labels.apply(axis=1, func=lambda row: not np.all(row==-1)) # -1 means unknown; return False if row consists of -1 only

		labels = labels.values[mask,:]
		features = features[mask,:]

		return features,labels

	return features

def split_data(features,labels=None,split_perc=0.8):

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
	if labels_file:
		xx,yy = load_data(features_file=features_file, labels_file=labels_file)
		x_train, y_train, x_test, y_test = split_data(xx, yy, split_perc)
	else:
		xx = load_data(features_file=features_file)
		x_train, x_test = split_data(xx, split_perc=split_perc)
		y_train,y_test = None,None

	smiles = DescriptorDataset(features=x_train, labels=y_train)
	smiles_test = DescriptorDataset(features=x_test, labels=y_test)

	dataloader = DataLoader(dataset=smiles, batch_size=batch_size, shuffle=True)
	dataloader_test = DataLoader(dataset=smiles_test, batch_size=len(smiles_test))

	return dataloader, dataloader_test