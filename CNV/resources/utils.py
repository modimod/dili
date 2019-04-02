import numpy as np
from utils.constants import tasks_idx, tasks, tasks_rank
from torch.nn import Sigmoid, LogSoftmax
import torch
from sklearn.metrics import confusion_matrix


class TensorList(list):

	def to(self, **kwargs):
		for i in range(len(self)):
			self[i] = self[i].to(**kwargs)

		return self


def transform_prediction_binary (y_pred, y_true, eval_col='vnctr'):
	# just take column of batch (column index of eval_col)
	idx = tasks_idx[eval_col]
	y_true = y_true[:, idx] if y_true.shape[1] > 1 else y_true[:,0]

	y_pred = y_pred[:, idx]

	# mask out -1 labels
	mask = (y_true != -1)
	y_true = y_true[mask]
	y_pred = y_pred[mask]

	if sum(mask) == 0:
		return list(), list(), list()

	# round to 0 or 1
	s = Sigmoid()
	y_pred = s(y_pred)

	return y_pred.round(), y_true, y_pred


def chain_predictions_binary(preds_list):
	return torch.cat(preds_list, dim=0)


def transform_prediction_classification (y_pred, y_true, eval_col='vnctr'):
	# take column of batch
	idx = tasks_idx[eval_col]
	y_true = y_true[:, idx] if len(y_true.shape) > 1 else y_true
	y_pred = y_pred[idx]  # in classification output is list

	# mask out -1 labels
	mask = (y_true != -1)

	if sum(mask) == 0:
		return list(), list(), list()

	y_true = y_true[mask]
	y_pred = y_pred[mask]

	# argmax to get predicted class
	s = LogSoftmax()
	y_pred = s(y_pred)
	y_pred = torch.argmax(y_pred, dim=1)

	return y_pred, y_true, list()


def chain_predictions_classification(preds):
	predictions = [list() for _ in range(len(tasks))]

	for a in preds:
		for p,ps in zip(predictions, a):
			p.extend(ps)

	return predictions


def transform_prediction_ranked (y_pred, y_true, eval_col='vnctr'):
	# take columns of batch that represent eval_col task
	idx = sum(list(tasks_rank.values())[:tasks_idx[eval_col]])
	idx_end = idx + tasks_rank[eval_col]
	y_true = y_true[:, idx:idx_end]

	y_pred = y_pred[:, idx:idx_end]

	# mask out -1 labels
	# label is vector with length tasks_rank[eval_col] (e.g. 3)
	# if label -1 - vector is [-1,-1,-1]
	# take first column for mask
	mask = (y_true[:,0] != -1)
	if sum(mask) == 0:
		return list(), list(), list()

	y_true = y_true[mask]
	y_pred = y_pred[mask]

	# sigmoid and round to 0 or 1
	s = Sigmoid()
	y_pred = s(y_pred)
	y_pred = y_pred.round()

	y_true_dtype = y_true.dtype
	y_pred_dtype = y_pred.dtype

	# get rightmost 1
	def get_rightmost_one(row):
		pos = np.argwhere(row == 1)
		return pos.max() + 1 if len(pos) > 0 else 0

	y_true = np.apply_along_axis(get_rightmost_one, axis=1, arr=y_true.cpu())
	y_pred = np.apply_along_axis(get_rightmost_one, axis=1, arr=y_pred.cpu())

	y_true = torch.from_numpy(y_true).to(dtype=y_true_dtype)
	y_pred = torch.from_numpy(y_pred).to(dtype=y_pred_dtype)

	return y_pred, y_true, list()

def labels_to_binary(labels, eval=False):

	if eval:
		return torch.where(labels > 0, torch.ones_like(labels), labels)

	# migrate dili annotations to binary
	labels_binary = torch.where(labels[:-1] > 0, torch.ones_like(labels[:-1]), labels[:-1])
	# migrate severity class to binary
	labels_sev = torch.where(labels[-1] > 4, torch.ones_like(labels[-1]), torch.zeros_like(labels[-1]))

	# add severity class and return
	return torch.cat([labels_binary, labels_sev.unsqueeze(dim=0)])


def labels_to_ranked(labels):
	def rank (v, max_val):
		if v.item() == -1:
			return torch.full(size=[max_val], fill_value=v)
		ranks = torch.zeros(max_val)
		ranks[: int(v.item())] = 1
		return ranks

	# migrate to ranked
	labels_ranked = [rank(v, t) for v, t in zip(labels, tasks_rank.values())]

	return torch.cat(labels_ranked).to(dtype=torch.float)


def transform_prediction_mse(y_pred, y_true, eval_col='vnctr'):
	# just take column of batch (column index of eval_col)
	idx = tasks_idx[eval_col]
	y_true = y_true[:, idx] if len(y_true.shape) > 1 else y_true

	y_pred = y_pred[:, idx]

	# mask out -1 labels
	mask = (y_true != -1)
	y_true = y_true[mask]
	y_pred = y_pred[mask]

	if sum(mask) == 0:
		return list(), list(), list()

	twos = torch.full(y_pred.size(),2).to(device=y_pred.device)
	zeros = torch.full(y_pred.size(),0).to(device=y_pred.device)

	y_pred_round = y_pred.round()
	y_pred_round = torch.where(y_pred_round > 2, twos, y_pred_round)
	y_pred_round = torch.where(y_pred_round < 0, zeros, y_pred_round)

	return y_pred_round, y_true, y_pred