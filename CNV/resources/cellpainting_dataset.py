import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class CellpaintingDataset(Dataset):

	"""Cellpainting dataset."""

	def __init__(self, csv_file, root_dir, file_ext, transform=None, mode_test=None):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			file_ext (string): File extension
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.data_file = pd.read_csv(csv_file, nrows=100 if mode_test else None)

		# for testing

		if mode_test:
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

	def get_labels(self):
		return self.data_file.iloc[:, 1:].values.astype(np.float)
