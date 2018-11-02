import torch.nn as nn
import torch.nn.functional as F
from net.modules.MultiOuts import MultiOuts


class CellpaintingCNV(nn.Module):
	def __init__(self, loss_functions = None, test_mode = None):
		super(CellpaintingCNV, self).__init__()
		self.conv1 = nn.Conv2d(5, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16*127*171, 120)
		self.fc2 = nn.Linear(120, 84)

		self.multiout = MultiOuts(84, loss_functions)

		self.loss = self.multiout.masked_loss

		self.test_mode = test_mode
		if test_mode:
			self.fc_test = nn.Linear(5*520*696, 84)

	def forward(self, x):
		if self.test_mode:
			return self.multiout(self.fc_test(x.view(-1, 5*520*696)))

		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16*127*171)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.multiout(x)
		return x

