from torch.utils.data import Dataset
import pandas as pd

class BaseDataset(Dataset):

	def __init__(self, csv_file, eval=None, transform=None):

		self.data_file = pd.read_csv(csv_file)

		try:
			self.clusters = self.data_file['cluster']
		except KeyError as e:
			print('no cluster column in dataset - I guess it is a test set')

		self.eval = eval
		self.transform = transform

		self.len = len(self.data_file)

		self.validation_fold = False

	def __len__(self):
		return self.len

	def validation(self):
		self.validation_fold = True

	def training(self):
		self.validation_fold = False

	def __getitem__(self, idx):
		pass