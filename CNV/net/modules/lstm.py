import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

from net.modules.MultiOuts import MultiOuts


class LSTMModule(nn.Module):
	def __init__ (self, input_dim, hidden_dim, num_layers=1, dropout=0.8):
		super(LSTMModule, self).__init__()
		self.hidden_dim = hidden_dim

		self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)#, bidirectional=True)
		self.dropout = nn.Dropout(p=dropout)

		self.multiout = MultiOuts(hidden_dim)

		self.loss = self.multiout.masked_loss

		#self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

	def forward (self, sentences, lengths):
		packed = pack_padded_sequence(sentences, lengths, batch_first=True)
		_, hidden = self.lstm(packed)
		hidden = self.dropout(hidden[0][-1]) # -1 means the output/hidden state of the last lstm layer

		outputs = self.multiout(hidden)
		#outputs = self.hidden2tag(hidden)
		#outputs = outputs.view(-1, self.tagset_size)
		#outputs = F.sigmoid(outputs)

		return outputs

	def forward_wo_packed (self, sentences):
		_, hidden = self.lstm(sentences[None,:,:])
		hidden = self.dropout(hidden[0][-1])  # -1 means the output/hidden state of the last lstm layer

		outputs = self.multiout(hidden)
		#outputs = self.hidden2tag(hidden)
		#outputs = outputs.view(-1, self.tagset_size)
		#outputs = F.sigmoid(outputs)
		return outputs

	#def masked_loss(self, outputs, targets, loss_function):
	#	mask = (targets != -1).to(dtype=torch.float)
	#	return loss_function(outputs * mask, targets * mask)

