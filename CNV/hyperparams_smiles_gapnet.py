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
	argparser.add_argument(r'-g', r'--gpus', type=str, help=r'GPU(s) to use, default empty (cpu)', required=False,
						   default='')
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

	settings.architecture.model_type = 'smiles_gap'
	print('Model type: {}'.format(settings.architecture.model_type))

	settings.log.log_dir = os.path.join(
		settings.log.log_dir,
		settings.data.label_format,
		settings.architecture.model_type)
	print('Log Dir: {}'.format(settings.log.log_dir))

	# for binary
	# params = [
	#	(0.0005, 128, 2, 0.5),
	#	(0.001, 16, 1, 0.0),
	#	(0.0005, 128, 2, 0.8)]

	#params = [(0.0005, 32, 1, 0.5), (0.0005, 32, 1, 0.0), (0.0005, 64, 1, 0.5)]

	# for classification
	#params = [(0.001, 32, 1, 0.0), (0.0005, 128, 1, 0.0), (0.001, 32, 1, 0.5)]

	#params = [(0.0005, 16, 1, 0.0), (0.001, 16, 1, 0.5), (0.001, 16, 1, 0.0)]

	# for ranked
	#params = [(0.001, 128, 1, 0.0), (0.001, 1024, 2, 0.0), (0.0005, 128, 2, 0.8)]

	params = [(0.001, 64, 1, 0.8), (0.001, 16, 1, 0.0), (0.001, 32, 1, 0.5)]

	lr_gap = [1e-5, 1e-5 * 5, 1e-4, 1e-4 * 5]
	fes = [True, *lr_gap]

	for p, fe in product(params, fes):
		settings.optimiser.learning_rate = p[0]
		settings.architecture.lstm_hidden_dim = p[1]
		settings.architecture.lstm_num_layers = p[2]
		settings.architecture.lstm_dropout = p[3]

		if fe is True:
			settings.architecture.feature_extract = fe
		else:
			settings.architecture.feature_extract = False
			settings.optimiser.learning_rate_gapnet = fe

		supervisor_class = supervisor_dict[settings.architecture.model_type]
		supervisor = supervisor_class(settings)
		perfs = supervisor.cross_validate()
