import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_

from sklearn.metrics import roc_auc_score,roc_curve,auc
import torch.nn.functional as F


def train (model, dataloader, optimizer, epochs, max_epochs=None, test_dataloader=None, norm_clip_value=None, verbose=None, save=None, early_stop=None, cuda=None):

	losses = list()
	losses_test = list()
	accs_test = list()
	auc_stats = list()

	num_batches = len(dataloader)

	best_model_loss = 1000
	best_model = None

	not_better = 0


	epochs = max_epochs if max_epochs else epochs
	for epoch in range(epochs):
		print()

		running_loss = 0.

		model.train()

		for i, (x, y) in enumerate(dataloader):

			if cuda:
				x = x.cuda()
				y = y.cuda()

			optimizer.zero_grad()

			preds = model(x)

			loss = model.loss(preds,y)

			loss.backward()

			if norm_clip_value is not None:
				clip_grad_norm_(model.parameters(), norm_clip_value)

			optimizer.step()

			if verbose:
				print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, i+1, num_batches, loss.item()))

			running_loss += loss.item()


		losses.append(running_loss/num_batches)
		print('epoch loss: {}'.format(running_loss/num_batches))

		# test on test set
		model.eval()

		with torch.no_grad():
			optimizer.zero_grad()

			dataiter = iter(test_dataloader)
			x_test, y_test = dataiter.next()
			if cuda:
				x_test = x_test.cuda()
				y_test = y_test.cuda()


			preds_test = model(x_test)

			loss_test = model.loss(preds_test,y_test,'vnctr')

			print('Epoch {}, Test Loss: {:.4f}'.format(epoch + 1, loss_test.item()))

			losses_test.append(loss_test.item())

			'''
			for i,(p,y_part) in enumerate(zip(preds_test,ys)):
				if i == 0:
					pred_label = F.sigmoid(p).round()

					y_part = y_part.contiguous()

					mask = (y_part.view(-1) != -1)

					total = int(mask.sum().item()) # count all true values

					correct = (pred_label.view(-1)[mask] == y_part.view(-1)[mask]).sum().item()

					if epoch % 10 == 0:
						macro_auc,fpr,tpr,roc_auc = get_auc_scores(y_part,F.sigmoid(p))
						print('macro auc: {:.4f}'.format(macro_auc))
						for k,v in roc_auc.items():
							print('auc class {}: {:.4f}'.format(k,v))

						auc_stats.append([macro_auc, fpr, tpr, roc_auc])

						accs_test.append(correct / total)

				else:
					pass

		print('Accuracy of the network on the {} test images after epoch {}: {:.2%}'.format(total, epoch+1, correct/total))
		'''

		# get best model
		if losses_test[-1] < best_model_loss:
			best_model_loss = losses_test[-1]
			best_model = model
			print('\n\nNew Best Model after Epoch {}!\n\n'.format(epoch))

		# early stopping
		if early_stop and epoch > 0:
			if losses_test[-1] > best_model_loss:
				not_better += 1
				print('No new best model for [{}/{}] epochs'.format(not_better,early_stop))

				if not_better >= early_stop:
					print('Training Finished')
					break
			else:
				not_better = 0

	if save:
		torch.save(best_model.state_dict(), save)

	return best_model, (losses,losses_test,accs_test, auc_stats)



def get_auc_scores(y_true,y_score):

	mask = (y_true.view(-1) != -1)
	macro_auc = roc_auc_score(y_true.view(-1)[mask],y_score.view(-1)[mask])

	batch_mask = (y_true != -1)

	# roc-curve & auc for every class
	fpr, tpr = dict(),dict()
	roc_auc = dict()

	for i in range(y_true.shape[1]):
		fpr[i], tpr[i], _ = roc_curve(y_true[batch_mask[:,i],i],y_score[batch_mask[:,i],i])
		roc_auc[i] = auc(fpr[i],tpr[i])


	return macro_auc, fpr,tpr,roc_auc

def get_accuracy (model, data):
	preds = list()
	labels = list()

	for i,(x,y) in enumerate(data):
		pred = model(x)

		label = 1 if pred.data.numpy()[0,0]>=0.5 else 0

		preds.append(label)
		labels.append(y.numpy())

	labels = np.array(labels,dtype=int)[:,0]
	preds = np.array(preds,dtype=int)

	acc = (labels == preds).sum() / len(data)

	return acc
