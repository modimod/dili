import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss,BCEWithLogitsLoss

class MultiOuts(nn.Module):
	def __init__(self, hidden_dim, tasks, loss_functions=None):
		super(MultiOuts, self).__init__()

		self.tasks = tasks
		self.loss_functions = list()

		if loss_functions is None:
			bce = BCEWithLogitsLoss()
			cel = CrossEntropyLoss()

			for _,t in tasks.items():
				if t == 1:	self.loss_functions.append(bce)
				else: self.loss_functions.append(cel)
		else:
			self.loss_functions = loss_functions

		self.hidden_to_out = nn.ModuleList()
		for _,t in tasks.items():
			self.hidden_to_out.append(nn.Linear(hidden_dim, t))

	def forward(self, input):
		outs = list()
		for l in self.hidden_to_out:
			outs.append(l(input))

		return outs

	def single_loss(self, output, target, loss):
		mask = (target != -1).to(dtype=torch.float)
		t = target * mask

		if type(loss).__name__ == 'BCEWithLogitsLoss':
			t = t.float()
			t = t.unsqueeze(1)
			mask = mask[:, None]
		elif type(loss).__name__ == 'CrossEntropyLoss':
			t = t.long()
			mask = mask[:, None]

		o = output * mask
		return loss(o, t)

	def masked_loss(self, outputs, targets, spec_target=None):

		total_loss = 0

		if spec_target is None:
			for o,t,l in zip(outputs,targets.transpose(0,1),self.loss_functions):
				total_loss += self.single_loss(o,t,l)
		else:
			idx = self.tasks[spec_target]
			total_loss = self.single_loss(outputs[idx],targets[:,idx],self.loss_functions[idx])

		return total_loss




class FC(nn.Module):
	#def __init__.py (self, input_dim, hidden_dim_1, hidden_dim_2, tagset_size_array, dropout=0.8):
	def __init__ (self, input_dim, hidden_dim_1, hidden_dim_2, tasks, loss_functions=None, dropout=0.8):
		super(FC, self).__init__()
		self.hidden_dim_1 = hidden_dim_1
		self.hidden_dim_2 = hidden_dim_2
		self.tasks = tasks

		self.dropout = nn.Dropout(p=dropout)

		self.fc1 = nn.Linear(input_dim, hidden_dim_1)
		self.selu = nn.SELU()
		self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)

		self.multiout = MultiOuts(hidden_dim_2, tasks, loss_functions)

		self.loss = self.multiout.masked_loss

	def forward (self, features):

		out = self.fc1(features)
		out = self.selu(out)
		out = self.fc2(out)
		out = self.selu(out)

		return self.multiout(out)

	#def loss(self, outputs, targets):
	#	return self.multiout.masked_loss(outputs,targets)