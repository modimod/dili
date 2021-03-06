from net.supervisors import DescrMSESupervisor
from settings import Settings
import argparse
import os
import torch
from itertools import product

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


	# log 06032019_descr_mse

	lrs = [0.0001, 0.001]
	hds = [
		[512, 256, 128, 64],
		[256, 128, 64, 32],
		[512, 256, 128],
		[256, 128, 64],
		[128, 64, 32],
		[512, 256],
		[256, 128],
		[128, 64],
		[64, 32],
		[256],
		[128],
		[64],
		[32],
		[16],
		[8]
		]
	dos = [0., 0.15]

	settings.log.log_dir = os.path.join(
		settings.log.log_dir,
		settings.data.label_format,
		settings.architecture.model_type)
	print('Log Dir: {}'.format(settings.log.log_dir))

	for lr, do, hd in product(lrs, dos, hds):

		settings.optimiser.learning_rate = lr
		settings.architecture.fc_hidden_dims = hd
		settings.architecture.fc_dropout = do

		supervisor = DescrMSESupervisor(settings)
		perfs = supervisor.cross_validate()
