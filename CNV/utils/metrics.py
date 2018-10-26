from utils.constants import tasks
import numpy as np
from sklearn.metrics import accuracy_score

class PerformanceEntry():

	def __init__(self):
		self.losses = list()
		self.accuracies = list()

	def update_loss(self, loss):
		self.losses.append(loss)

	def update_accuracy(self, acc):
		self.accuracies.append(acc)

	def calc_accuracy(self, predictions, labels):
		# TODO calc
		idx = tasks['vnctr']

		predictions = np.array(predictions)
		predictions = predictions[:, idx]

		labels = labels[:, idx]

		acc = accuracy_score(y_true=labels, y_pred=predictions)

		self.accuracies.append(acc)

	def __str__(self):
		print('Accs: {} \nLosses: {}'.format(self.accuracies, self.losses))