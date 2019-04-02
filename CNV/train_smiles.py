from net.supervisors import SmilesSupervisor
from resources.smiles_dataset import SmilesBinaryDS
from settings import Settings
import argparse
import os
import torch
from torch.utils.data import DataLoader
from resources.transforms import Normalize

if __name__=='__main__':
	torch.backends.cudnn.benchmark = True

	argparser = argparse.ArgumentParser(description='Train DILI model')

	argparser.add_argument(r'-s', r'--settings', type=str, help=r'settings file to use', required=True)
	argparser.add_argument(r'-g', r'--gpus', type=str, help=r'GPU(s) to use, default empty (cpu)', required=False, default='')
	argparser.add_argument(r'-lf', r'--labelformat', type=str, help=r'label format', required=True)

	args = argparser.parse_args()

	print('GPUs: {}'.format(args.gpus))
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

	# get settings
	settings_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'settings', args.settings)
	settings = Settings.from_json_file(settings_file)

	settings.run.device = 'cuda' if torch.cuda.is_available() else 'cpu'

	print('Used device: {}'.format(settings.run.device))

	settings.data.label_format = args.labelformat

	settings.architecture.model_type = 'descr'
	print('Model type: {}'.format(settings.architecture.model_type))

	settings.log.log_dir = os.path.join(
		settings.log.log_dir,
		settings.data.label_format,
		settings.architecture.model_type,
		'trained'
		)
	print('Log Dir: {}'.format(settings.log.log_dir))

	params = (0.0005, 32, 1, 0.5)

	settings.optimiser.learning_rate = params[0]
	settings.architecture.lstm_hidden_dim = params[1]
	settings.architecture.lstm_num_layers = params[2]
	settings.architecture.lstm_dropout = params[3]

	supervisor = SmilesSupervisor(settings)

	supervisor.train_plain(epochs=50)

	# get dataset
	testset = SmilesBinaryDS(csv_file=settings.data.csv_file_test)#, eval=True)

	#mean, std = testset.get_mean_std()
	#transform = Normalize(mean, std)

	#testset.transform = transform
	testset.transform = supervisor.dataset.transform
	testloader = DataLoader(dataset=testset, batch_size=settings.run.batch_size_eval, shuffle=True, collate_fn=testset.collate_fn)

	perf = supervisor._evaluate(testloader, 0)
	supervisor._save_performance(perf, 'test_perf')

	print(perf.to_dictionary())





