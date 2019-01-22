from sklearn.metrics import roc_auc_score

import numpy as np

from typing import Iterator, List, Tuple, Union
from collections import OrderedDict


class PerformanceEntry(object):
	"""
	Class responsible for logging and fetching performance metrics.
	"""

	def __init__(self, num_classes: int) -> None:
		"""
		Initialise performance logger (multi-class problem). All metrics are computed according to the
		macro-averaging approach, so the bigger classes are not dominating the smaller ones [1].

		[1] http://rali.iro.umontreal.ca/rali/sites/default/files/publis/SokolovaLapalme-JIPM09.pdf

		:param num_classes: amount of different classes to be supported
		"""
		if num_classes < 2:
			raise ValueError(r'There need to be at least 2 different classes ({} provided)!'.format(num_classes))

		self.__num_classes = num_classes
		self.__loss = None
		self.__cm = None
		self.__auc = None # only for binary
		self.reset()

	def better_than(self, other):
		# its better when loss is small
		return self.loss < other.loss if len(other.__loss) > 0 else True

	def __iadd__(self, other: r'PerformanceEntry') -> r'PerformanceEntry':
		"""
		Return the current performance logger with an updated internal state.

		:param other: array-like object representing a confusion matrix of the same size as this instance's one
		:return: updated performance logger using the current instance
		"""
		if not other.confusion_matrix.shape == self.__cm.shape:
			raise ValueError(r'Shape mismatch: {} provided, {} expected!'.format(
				other.confusion_matrix.shape, self.__cm.shape))

		self.__cm += other.confusion_matrix
		self.__loss.extend(other.__loss)
		return self

	def update_confusion_matrix(self, other: np.ndarray) -> None:
		"""
		Return the current performance logger with an updated internal state.

		:param other: array-like object representing a confusion matrix of the same size as this instance's one
		"""
		if not other.shape == self.__cm.shape:
			raise ValueError(r'Shape mismatch: {} provided, {} expected!'.format(other.shape, self.__cm.shape))

		try:
			other = other.astype(dtype=np.uint32)
		except ValueError:
			raise ValueError(r'Type mismatch: {} not compatible with uint32!'.format(other.dtype))

		self.__cm += other

	def update_loss(self, other: Union[float, np.float32]) -> None:
		"""
		Append loss to the internal collection of losses.

		:param other: loss to append
		"""
		self.__loss.append(other)

	def _split_confusion_matrix(self) -> Iterator[Tuple[np.uint32]]:
		"""
		Split confusion matrix into TP, FN, FP and TN. It is assumed, that the underlying confusion matrix has the
		predictions as columns, and true labels as rows.

		:return: TP, FN, FP and TN for current class
		"""
		for current_class in range(self.__num_classes):
			tp = self.__cm[current_class, current_class].astype(np.uint32)
			fn = (self.__cm[current_class].sum() - tp).astype(np.uint32)
			fp = (self.__cm[:, current_class].sum() - tp).astype(np.uint32)
			tn = (self.__cm.sum() - (tp + fp + fn)).astype(np.uint32)
			yield tp, fn, fp, tn

	def to_dictionary(self) -> dict:
		"""
		Transforms the current performance logger instance to a dictionary, containing all performance metrics.

		:return: dictionary of current performance logger instance, containing all performance metrics
		"""
		return OrderedDict({
			r'loss': self.loss.astype(float),
			r'confusion_matrix': self.confusion_matrix.tolist(),
			r'num_class': self.num_classes,
			r'accuracy': self.accuracy.astype(float),
			r'accuracy_per_class': [_.astype(float) for _ in self.accuracy_per_class],
			r'balanced_accuracy': self.balanced_accuracy.astype(float),
			r'balanced_accuracy_per_class': [_.astype(float) for _ in self.balanced_accuracy_per_class],
			r'precision': self.precision.astype(float),
			r'precision_per_class': [_.astype(float) for _ in self.precision_per_class],
			r'recall': self.recall.astype(float),
			r'recall_per_class': [_.astype(float) for _ in self.recall_per_class],
			r'specificity': self.specificity.astype(float),
			r'specificity_per_class': [_.astype(float) for _ in self.specificity_per_class],
			r'f1score': self.f1score.astype(float),
			r'f1score_per_class': [_.astype(float) for _ in self.f1score_per_class],
			r'auc_score': self.__auc
		})

	def reset(self) -> None:
		"""
		Reset internal loss collecion and confusion matrix.
		"""
		self.__loss = []
		self.__cm = np.zeros(shape=(self.__num_classes, self.__num_classes), dtype=np.uint32)

	@property
	def loss(self) -> np.float32:
		return np.average(self.__loss.copy()).astype(np.float32)

	@property
	def accuracy(self) -> np.float32:
		return np.float32(self.__cm.diagonal().sum() / self.__cm.sum())

	@property
	def accuracy_per_class(self) -> List[np.float32]:
		return [np.float32(np.divide(tp + tn, tp + fn + fp + tn, out=np.zeros(shape=[]), where=(tp + fn + fp + tn) > 0))
				for tp, fn, fp, tn in self._split_confusion_matrix()]

	@property
	def balanced_accuracy(self) -> np.float32:
		return np.average(self.balanced_accuracy_per_class)

	@property
	def balanced_accuracy_per_class(self) -> List[np.float32]:
		return [(recall + specificity) / 2
				for recall, specificity in zip(self.recall_per_class, self.specificity_per_class)]

	@property
	def precision(self) -> np.float32:
		return np.average(self.precision_per_class)

	@property
	def precision_per_class(self) -> List[np.float32]:
		return [np.float32(np.divide(tp, tp + fp, out=np.zeros(shape=[]), where=(tp + fp) > 0))
				for tp, fn, fp, tn in self._split_confusion_matrix()]

	@property
	def recall(self) -> np.float32:
		return np.average(self.recall_per_class)

	@property
	def recall_per_class(self) -> List[np.float32]:
		return [np.float32(np.divide(tp, tp + fn, out=np.zeros(shape=[]), where=(tp + fn) > 0))
				for tp, fn, fp, tn in self._split_confusion_matrix()]

	@property
	def specificity(self) -> np.float32:
		return np.average(self.specificity_per_class)

	@property
	def specificity_per_class(self) -> List[np.float32]:
		return [np.float32(np.divide(tn, tn + fp, out=np.zeros(shape=[]), where=(tn + fp) > 0))
				for tp, fn, fp, tn in self._split_confusion_matrix()]

	@property
	def f1score(self) -> np.float32:
		return np.average(self.f1score_per_class)

	@property
	def f1score_per_class(self) -> List[np.float32]:
		return [np.float32(np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(shape=[]), where=(2 * tp + fp + fn) > 0))
				for tp, fn, fp, tn in self._split_confusion_matrix()]

	@property
	def num_classes(self) -> int:
		return self.__num_classes

	@property
	def confusion_matrix(self) -> np.ndarray:
		return self.__cm

	@property
	def auc(self):
		return self.__auc

	def set_auc_scores (self, y_true, y_score):
		self.__auc = roc_auc_score(y_true=y_true, y_score=y_score)

