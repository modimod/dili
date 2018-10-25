from FC import FC
from Descriptor_Dataset import DescriptorDataset, get_loader, load_data, split_data
import torch.nn.functional as F
import torch
from torch.nn import BCELoss,CrossEntropyLoss,BCEWithLogitsLoss
from torch.optim import Adam
from train_FC import train
import matplotlib.pyplot as plt

from collections import OrderedDict



features_file = '/Users/Moerti/AA_projects/DILI/data/smiles_descriptors_rdkit_normalized.csv'
labels_file = '/Users/Moerti/AA_projects/DILI/data/dili_dataset_4_labels.csv'

dl,dl_test = get_loader(features_file,labels_file=labels_file,batch_size=32)

bce = BCEWithLogitsLoss()
cel = CrossEntropyLoss()

input_dim = dl.dataset.features.shape[1]
hidden_dim_1 = 32
hidden_dim_2 = 32
tasks = OrderedDict([
	('sakatis',1),('zhu',1),('xu',1),('greene',3),('vnctr',4),('nctr',3),('severity',9)])
#tasks = [1,1,1,3,4,3,9]

model = FC(input_dim=input_dim,hidden_dim_1=hidden_dim_1,hidden_dim_2=hidden_dim_2,tasks=tasks,loss_functions=None)

optimizer = Adam(model.parameters(),lr=1e-4)

epochs = 10
max_epochs = 100
early_stop = 5
norm_clip_value = None

model, (losses,losses_test,accs_test, auc_stats) = train(model=model,
											  dataloader=dl,
											  optimizer=optimizer,
											  epochs=epochs,
											  max_epochs=max_epochs,
											  test_dataloader=dl_test,
											  early_stop=early_stop,
											  norm_clip_value=norm_clip_value,
											  save=None,
											  cuda=False,
											  verbose=False)

plt.plot(losses,label='train')
plt.plot(losses_test,label='valid')
plt.legend()
plt.show()