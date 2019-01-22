from net.supervisors import SmilesCellpaintingSupervisor, SmilesSupervisor, DescrCellpaintingSupervisor, DescrSupervisor
from settings import Settings
import argparse
import os
import torch
from itertools import product

supervisor_dict = {
	'descr': DescrSupervisor,
	'descr_gap': DescrCellpaintingSupervisor,
	'smiles': SmilesSupervisor,
	'smiles_gap': SmilesCellpaintingSupervisor
}

if __name__=='__main__':


	torch.backends.cudnn.benchmark = True

	argparser = argparse.ArgumentParser(description='Train DILI model')

	argparser.add_argument(r'-s', r'--settings', type=str, help=r'settings file to use', required=True)
	argparser.add_argument(r'-d', r'--device', type=str, help=r'device to use (e.g. cuda:0)', required=True)
	argparser.add_argument(r'-lf', r'--labelformat', type=str, help=r'label format', required=True)

	args = argparser.parse_args()

	# get settings
	settings_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'settings', args.settings)
	settings = Settings.from_json_file(settings_file)

	settings.run.device = args.device
	print('Used device: {}'.format(settings.run.device))

	settings.data.label_format = args.labelformat

	settings.architecture.model_type = 'smiles'
	print('Model type: {}'.format(settings.architecture.model_type))

	settings.log.log_dir = os.path.join(
		settings.log.log_dir,
		settings.data.label_format,
		settings.architecture.model_type)
	print('Log Dir: {}'.format(settings.log.log_dir))

	lrs = [0.0001, 0.0005, 0.001]
	hds = [16, 32, 64, 128, 256, 512, 1024]
	nls = [1, 2]
	dos = [0., 0.5, 0.8]


	for lr, hd, nl, do in product(lrs, hds, nls, dos):
		settings.optimiser.learning_rate = lr
		settings.architecture.lstm_hidden_dim = hd
		settings.architecture.lstm_num_layers = nl
		settings.architecture.lstm_dropout = do

		supervisor_class = supervisor_dict[settings.architecture.model_type]
		supervisor = supervisor_class(settings)
		perfs = supervisor.cross_validate()
