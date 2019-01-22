import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss,BCEWithLogitsLoss, MSELoss
from utils.constants import tasks, tasks_idx, pandas_cols

class MultiOuts(nn.Module):
	def __init__(self, hidden_dim, loss_functions=None):
		super().__init__()

		# TODO get rid of loss_function argument

		self.tasks = tasks
		self.loss_functions = list()

		if loss_functions is None:
			bce = BCEWithLogitsLoss()
			cel = CrossEntropyLoss()
			mse = MSELoss()

			for ann,t in tasks.items():
				if ann in ['sakatis', 'zhu', 'xu']:
					self.loss_functions.append(bce)
				else:
					self.loss_functions.append(cel)

			# add severity class
			self.loss_functions.append(mse)
		else:
			self.loss_functions = loss_functions

		self.hidden_to_out = nn.ModuleList()
		for _,t in tasks.items():
			self.hidden_to_out.append(nn.Linear(hidden_dim, t))

		# add severity class
		self.hidden_to_out.append(nn.Linear(hidden_dim, 1))

	def forward(self, x):
		outs = list()
		for l in self.hidden_to_out:
			outs.append(l(x))

		return outs

	def single_loss(self, output, target, loss):
		mask = (target != -1).to(dtype=target.dtype)
		t = target * mask

		if type(loss).__name__ == 'BCEWithLogitsLoss':
			t = t.float()
			t = t.unsqueeze(1)
			mask = mask[:, None]
		elif type(loss).__name__ == 'CrossEntropyLoss':
			t = t.long()
			mask = mask[:, None]
		else:  						# MSELoss
			output = torch.squeeze(output)

		mask = mask.to(dtype=output.dtype)
		o = output * mask
		return loss(o, t)

	def masked_loss(self, outputs, targets, spec_target=None):

		total_loss = 0

		if spec_target is None:
			for o,t,l in zip(outputs,targets.transpose(0,1),self.loss_functions):
				total_loss += self.single_loss(o,t,l)
		else:
			idx = tasks_idx[spec_target]

			targets = targets[:,idx] if len(targets.shape) > 1 else targets

			total_loss = self.single_loss(outputs[idx], targets, self.loss_functions[idx])

		return total_loss


class MultiOutsBinary(nn.Module):

	def __init__(self, hidden_dim, dili_out_dim):
		super().__init__()

		self.hidden_to_out = nn.ModuleList()

		# dili annotations
		self.hidden_to_out.append(nn.Linear(hidden_dim, dili_out_dim))

		# severity class
		self.hidden_to_out.append(nn.Linear(hidden_dim, 1))

	def forward(self, x):
		outs = list()
		for l in self.hidden_to_out:
			outs.append(l(x))

		return outs

	def loss(self, outputs, targets, spec_target=None):
		'''

		:param outputs: (list) [dili_binary_block, severity_class]
		:param targets:
		:param spec_target:
		:return:
		'''
		total_loss = masked_loss_binary(outputs[0], targets[:,:-1], spec_target)
		sev_loss = single_loss(outputs[1], targets[:,-1], MSELoss())

		print(total_loss.item(), sev_loss.item(), sev_loss.item()*0.001/total_loss.item() * 100)

		total_loss += sev_loss

		return total_loss


def masked_loss_binary(outputs, targets, spec_target=None):
	loss = BCEWithLogitsLoss()

	if spec_target:
		idx = tasks_idx[spec_target]
		targets = targets[:, idx] if len(targets.shape) > 1 else targets
		outputs = outputs[:, idx]

	mask = (targets != -1).to(dtype=torch.float)
	outputs = outputs * mask
	targets = targets * mask

	return loss(outputs, targets)

def single_loss(output, target, loss):
	mask = (target != -1).to(dtype=target.dtype)
	t = target * mask

	if type(loss).__name__ == 'BCEWithLogitsLoss':
		t = t.float()
		t = t.unsqueeze(1)
		mask = mask[:, None]
	elif type(loss).__name__ == 'CrossEntropyLoss':
		t = t.long()
		mask = mask[:, None]
	else:  						# MSELoss
		output = torch.squeeze(output)

	mask = mask.to(dtype=output.dtype)
	o = output * mask
	return loss(o, t)
