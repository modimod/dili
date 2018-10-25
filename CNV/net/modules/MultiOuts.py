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

