import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch


class CellpaintingDataset(Dataset):

	"""Cellpainting dataset."""

	def __init__(self, csv_file, root_dir, file_ext, transform=None):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			file_ext (string): File extension
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.data_file = pd.read_csv(csv_file)

		# for testing
		self.data_file['SAMPLE_KEY'] = '26247-P17-5'

		assert self.data_file['SAMPLE_KEY'].apply(lambda x: os.path.isfile(os.path.join(root_dir, x) + file_ext)).all(), \
			"Some images referenced in the CSV file were not found"

		self.root_dir = root_dir
		self.file_ext = file_ext
		self.transform = transform
		self.len = len(self.data_file)

	def __len__(self):
		return self.len

	def __getitem__(self, idx):
		img_name = os.path.join(self.root_dir, '{}.npz'.format(self.data_file.iloc[idx, 0]))
		#image = io.imread(img_name)
		image = np.load(img_name)
		image = image['sample'].astype(np.float)

		labels = self.data_file.iloc[idx, 1:].values.astype(np.float)

		sample = {'image': image, 'labels': labels}

		if self.transform:
			sample = self.transform(sample)

		return sample


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

	images = CellpaintingDataset(features=x_train, labels=y_train)
	images_test = CellpaintingDataset(features=x_test, labels=y_test)

	dataloader = DataLoader(dataset=images, batch_size=batch_size, shuffle=True)
	dataloader_test = DataLoader(dataset=images_test, batch_size=len(images_test))

	return dataloader, dataloader_test
