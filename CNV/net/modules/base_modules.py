import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from math import sqrt


class LSTMModule(nn.Module):
	def __init__ (self, input_dim, hidden_dim, num_layers=1, dropout=0.8):
		super(LSTMModule, self).__init__()
		self.hidden_dim = hidden_dim

		self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)#, bidirectional=True)
		self.dropout = nn.Dropout(p=dropout)

	def forward (self, sentences, lengths):
		packed = pack_padded_sequence(sentences, lengths, batch_first=True)
		_, hidden = self.lstm(packed)
		hidden = self.dropout(hidden[0][-1]) # -1 means the output/hidden state of the last lstm layer

		return hidden


class FCModule(nn.Module):
	def __init__ (self, input_dim, hidden_dim_1, hidden_dim_2, dropout=0.8):
		super().__init__()

		self.classifier = nn.Sequential(
			fc_block(input_dim, hidden_dim_1, dropout),
			fc_block(hidden_dim_1, hidden_dim_2, dropout)
		)

		self.selu_init()

	def forward(self, features):
		return self.classifier(features)

	def selu_init(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				fan_in = m.in_features
				nn.init.normal_(m.weight, 0, sqrt(1. / fan_in))

class FCDynamicModule(nn.Module):
	def __init__(self, input_dim, hidden_dims, dropout=0.8):
		super().__init__()

		self.sizes = [input_dim, *hidden_dims]

		fc_blocks = [fc_block(in_f, out_f, dropout) for in_f,out_f in zip(self.sizes, self.sizes[1:])]

		self.classifier = nn.Sequential(*fc_blocks)

		self.selu_init()

	def forward(self, features):
		return self.classifier(features)

	def selu_init(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				fan_in = m.in_features
				nn.init.normal_(m.weight, 0, sqrt(1. / fan_in))


def fc_block(in_f, out_f, dropout):
	return nn.Sequential(
		nn.Linear(in_f, out_f),
		nn.SELU(inplace=True),
		nn.AlphaDropout(p=dropout)
	)
