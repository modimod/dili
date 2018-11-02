from utils.constants import tasks,tasks_idx
import numpy as np
from sklearn.metrics import accuracy_score
import torch

class PerformanceEntry():

	def __init__(self):
		self.loss = None
		self.acc = None

	def calc_accuracy(self, predictions, labels):
		# TODO calc
		idx = tasks_idx['vnctr']

		preds = predictions[idx]
		preds = torch.stack(preds).cpu().numpy()
		preds = np.argmax(preds, axis=1)

		labels = labels[:, idx]

		self.acc = accuracy_score(y_true=labels, y_pred=preds)

	# @property
	# def loss(self):
	# 	return self.loss
	#
	# @loss.setter
	# def loss(self, new_loss):
	# 	self._loss = new_loss
	#
	# @property
	# def acc(self):
	# 	return self.acc
	#
	# @acc.setter
	# def acc(self, new_acc):
	# 	self._acc = new_acc





	def __str__(self):
		print('Accs: {} \nLosses: {}'.format(self.accuracies, self.losses))


