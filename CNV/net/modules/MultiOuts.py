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

			for ann,t in tasks.items():
				if ann in ['sakatis', 'zhu', 'xu']:
					self.loss_functions.append(bce)
				else:
					self.loss_functions.append(cel)
		else:
			self.loss_functions = loss_functions

		self.hidden_to_out = nn.ModuleList()
		for _,t in tasks.items():
			self.hidden_to_out.append(nn.Linear(hidden_dim, t))

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

		mask = mask.to(dtype=output.dtype)
		o = output * mask
		return loss(o, t)

	def masked_loss(self, outputs, targets, spec_target=None):

		total_loss = 0

		#losses = list()

		if spec_target is None:
			for o,t,l in zip(outputs,targets.transpose(0,1),self.loss_functions):
				#losses.append(self.single_loss(o,t,l))
				#total_loss = sum(losses)
				total_loss += self.single_loss(o,t,l)
			#print(["{0:0.2f}".format(i.item()) for i in losses])
		else:
			idx = tasks_idx[spec_target]

			targets = targets[:,idx] if len(targets.shape) > 1 else targets

			total_loss = self.single_loss(outputs[idx], targets, self.loss_functions[idx])

		return total_loss


class MultiOutsBinary(nn.Module):

	def __init__(self, hidden_dim, out_dim):
		super().__init__()

		self.fc = nn.Linear(hidden_dim, out_dim)

		self.loss = masked_loss_binary

	def forward(self, x):
		return self.fc(x)


def masked_loss_binary(outputs, targets, spec_target=None):
	loss = BCEWithLogitsLoss()

	if spec_target:
		idx = tasks_idx[spec_target]
		targets = targets[:, idx] if targets.shape[1] > 1 else targets[:,0]
		outputs = outputs[:, idx]

	mask = (targets != -1).to(dtype=torch.float)
	outputs = outputs * mask
	targets = targets * mask

	return loss(outputs, targets)

class MultiOutsMSE(nn.Module):

	def __init__(self, hidden_dim, out_dim):
		super().__init__()

		self.fc = nn.Linear(hidden_dim, out_dim)

		self.loss = masked_loss_mse

	def forward(self, x):
		return self.fc(x)

def masked_loss_mse(outputs, targets, spec_target=None):
	loss = MSELoss()

	if spec_target:
		idx = tasks_idx[spec_target]
		targets = targets[:, idx] if len(targets.shape) > 1 else targets
		outputs = outputs[:, idx]

	mask = (targets != -1).to(dtype=torch.float)
	outputs = outputs * mask
	targets = targets * mask

	return loss(outputs, targets)