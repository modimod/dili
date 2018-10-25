

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
		acc = 0.8

		self.accuracies.append(acc)

	def __str__(self):
		print(self.accuracies)