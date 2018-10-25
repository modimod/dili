import argparse
from os import makedirs,path
from Dataset_MultitaskMask import get_loader,load_data,SmilesDataset,DataLoader,collate_fn
from train_multitask_mask import train
from torch.nn import BCELoss
from torch.optim import Adam
from LSTMTagger import LSTMTaggerMultitaskMask
import torch
import time
import numpy as np
import pandas as pd
import os

parser = argparse.ArgumentParser(description='train lstm for multitask sequence classification')
parser.add_argument('hidden_dim', metavar='hidden_dim', type=int, default=64, help='hidden dimension of the lstm cell')
parser.add_argument('lstm_layers', metavar='lstm_layers', type=int, default=1, help='number of layers of the lstm cell')
parser.add_argument('max_epochs', metavar='max_epochs', type=int, default=100, help='max number of epochs to train')
#parser.add_argument('epochs', metavar='epochs', type=int, default=10, help='the number of epochs (overwritten if max_epochs is given')
parser.add_argument('early_stop', metavar='early_stop', type=int, default=3, help='the early stopping criterion')
parser.add_argument('learning_rate', metavar='learning_rate', type=float, default=1e-3, help='the learning_rate')
parser.add_argument('batch_size', metavar='batch_size', type=int, default=32, help='batch size')
parser.add_argument('norm_clip_value', metavar='norm_clip_value', type=int, default=5, help='weight norm clip value')
parser.add_argument('dropout', metavar='dropout', type=float, default=0.5, help='dropout value')
parser.add_argument('sliding_window', metavar='sliding_window', type=int, default=1, help='slding window size')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='activate CUDA training')

#args = parser.parse_args('64 2 1 3 1e-3 32 5 0.5 3 --cuda'.split())
args = parser.parse_args()

# check if cuda is available
args.cuda = args.cuda and torch.cuda.is_available()

time_str = str(int(round(time.time() * 1000)))

curr_path = os.path.dirname(os.path.abspath(__file__))

time_dir = os.path.join(curr_path,'..','saves',time_str)

hidden_dim = args.hidden_dim #64
lstm_layers = args.lstm_layers #2
tagset_size = 9
epochs = 1
learning_rate = args.learning_rate #1e-2
batch_size = args.batch_size #32
norm_clip_value = args.norm_clip_value #5
dropout = args.dropout #0.5
early_stop = args.early_stop #3
max_epochs = args.max_epochs #100
sliding_window = args.sliding_window #1

alphabet = '#%()+-./0123456789=@ABCFGHIKLMNOPRSTVZ[\\]abcdeghilnorstu'

params_txt = 'hidden_dim: {}\nnum_layers: {}\n' \
			 'max_epochs: {}\nearly_stop: {}\n' \
			 'learning_rate: {}\nbatch_size: {}\n' \
			 'norm_clip_value: {}\ndropout: {}\n' \
			 'sliding_window: {}\ntagset_size: {}\n'.format(hidden_dim, lstm_layers, max_epochs, early_stop, learning_rate, batch_size, norm_clip_value, dropout, sliding_window, tagset_size)


data_path = os.path.join(curr_path,'..','AILS_Challenge')

ajoin = os.path.join

dataloader,dataloader_test = get_loader(features_file=ajoin(data_path,'chem_train.csv'),labels_file=ajoin(data_path,'y_train.csv'),batch_size=batch_size,sliding_window=sliding_window)
model = LSTMTaggerMultitaskMask(input_dim=sliding_window*len(alphabet),hidden_dim=hidden_dim,tagset_size=tagset_size, num_layers=lstm_layers,dropout=dropout)

if args.cuda:
	model = model.cuda()

if args.cuda:
	loss = BCELoss().cuda()
else:
	loss = BCELoss()

optimizer = Adam(model.parameters(), lr=learning_rate)

print('Start training of Multitask Learning with following (hyper)parameters:')
print('CUDA: {}'.format(args.cuda))
print('Folder: {}'.format(time_dir))
print('Loss Function: {}'.format(type(loss).__name__))
print('Optimizer: {}'.format(type(optimizer).__name__))
print('Max Epochs: {}'.format(max_epochs))
print('Early Stopping: {}'.format(early_stop))
print('Norm Clipping Value: {}'.format(norm_clip_value))
print('Learning Rate: {}'.format(learning_rate))
print('Hidden Dimension: {}'.format(hidden_dim))
print('LSTM Layers: {}'.format(lstm_layers))
print('Dropout: {}'.format(dropout))
print('Sliding Window: {}'.format(sliding_window))
print('Batch Size: {}'.format(batch_size))
print()



# create directory
makedirs(time_dir)

model, (losses,losses_test,accs_test, auc_stats) = train(model=model,
											  dataloader=dataloader,
											  loss_function=loss,
											  optimizer=optimizer,
											  epochs=epochs,
											  max_epochs=max_epochs,
											  test_dataloader=dataloader_test,
											  early_stop=early_stop,
											  norm_clip_value=norm_clip_value,
											  save=path.join(time_dir,'best_model.ckpt'),
											  cuda=args.cuda,
											  verbose=None)

print('finished training')
print('saving stats...')


# save loss-arrays

np.savetxt(path.join(time_dir,'losses_train.csv'), losses)
np.savetxt(path.join(time_dir,'losses_test.csv'), losses_test)
np.savetxt(path.join(time_dir,'accs_test.csv'), accs_test)

macro_auc = list()
class_auc = list()
for i in range(len(auc_stats)):
	macro_auc.append(auc_stats[i][0])
	class_auc.append([v for k,v in auc_stats[i][3].items()])

np.savetxt(path.join(time_dir,'macro_auc.csv'),np.array(macro_auc))
np.savetxt(path.join(time_dir,'class_auc.csv'),np.array(class_auc))

params_txt += 'macro aucs last epoch: {:.4f}'.format(macro_auc[-1])

with open(path.join(time_dir,'parameters.txt'),'w') as file:
	file.write(params_txt)

print('saved!')
print('folder name: {}'.format(time_dir))



def eval_set(model, file, outfile):

	xx = load_data(features_file=file)


	model.eval()
	with torch.no_grad():

		ds = SmilesDataset(features=xx, sliding_window=sliding_window)

		preds = np.zeros((len(ds),tagset_size))

		for i in range(len(ds)):
			x = ds[i]

			if args.cuda:
				x = x.cuda()

			pred = model.forward_wo_packed(x)

			if args.cuda:
				pred = pred.cpu()

			pred = pred.numpy()

			preds[i] = pred


	cols = 'Task1 Task2 Task3 Task4 Task5 Task6 Task7 Task8 Task9'.split()

	preds = pd.DataFrame(preds, columns=cols, dtype=float)

	preds.to_csv(outfile, index=False, float_format='%.3f')








### classify leaderboard data

print('start classify leaderboard')
eval_set(model, ajoin(data_path,'chem_leaderboard.csv'), os.path.join(time_dir,'predicted_labels_leaderboard.csv'))
print('classify leaderboard finished')

### classify test data

print('start classify test')
eval_set(model, ajoin(data_path,'chem_test.csv'), os.path.join(time_dir,'predicted_labels_test.csv'))
print('classify test finished')


print('folder name: {}'.format(time_dir))
