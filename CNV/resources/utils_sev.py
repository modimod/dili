import numpy as np
from utils.constants import tasks_idx, tasks, tasks_rank
from torch.nn import Sigmoid
import torch


def transform_prediction_binary (y_pred, y_true, eval_col='vnctr'):
	# just take column of batch (column index of eval_col)
	idx = tasks_idx[eval_col]
	y_true = y_true[:, idx] if len(y_true.shape) > 1 else y_true

	# only take dili-block (get rid of severity class block)
	y_pred = y_pred[0]

	y_pred = y_pred[:, idx]

	# mask out -1 labels
	mask = (y_true != -1)
	y_true = y_true[mask]
	y_pred = y_pred[mask]

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
		return list(), list()

	y_true = y_true[mask]
	y_pred = y_pred[mask]

	# argmax to get predicted class
	y_pred = np.argmax(y_pred, axis=1)

	return y_pred, y_true


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

	# only take dili-block (get rid of severity class block)
	y_pred = y_pred[0]

	y_pred = y_pred[:, idx:idx_end]

	# mask out -1 labels
	# label is vector with length tasks_rank[eval_col] (e.g. 3)
	# if label -1 - vector is [-1,-1,-1]
	# take first column for mask
	mask = (y_true[:,0] != -1)
	if sum(mask) == 0:
		return list(), list()

	y_true = y_true[mask]
	y_pred = y_pred[mask]

	# sigmoid and round to 0 or 1
	s = Sigmoid()
	y_pred = s(y_pred)
	y_pred = y_pred.round()

	# get rightmost 1
	def get_rightmost_one(row):
		pos = np.argwhere(row == 1)
		return pos.max() + 1 if len(pos) > 0 else 0

	y_true = np.apply_along_axis(get_rightmost_one, axis=1, arr=y_true)
	y_pred = np.apply_along_axis(get_rightmost_one, axis=1, arr=y_pred)

	return y_pred, y_true

def labels_to_binary(labels):

	# migrate dili annotations to binary
	labels_binary = torch.where(labels[:-1] > 0, torch.ones_like(labels[:-1]), labels[:-1])

	# add severity class and return
	return torch.cat([labels_binary, labels[-1].unsqueeze(dim=0)])


def labels_to_ranked(labels):
	def rank (v, max_val):
		if v.item() == -1:
			return torch.full(size=[max_val], fill_value=v)
		ranks = torch.zeros(max_val)
		ranks[: int(v.item())] = 1
		return ranks

	# migrate dili annotations to ranked (not severity class)
	labels_ranked = [rank(v, t) for v, t in zip(labels[:-1], tasks_rank.values())]

	# add severity class
	labels_ranked.append(labels[-1].unsqueeze(dim=0))

	return torch.cat(labels_ranked).to(dtype=torch.float)

