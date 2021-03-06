from net.supervisors import SmilesCellpaintingSupervisor, SmilesSupervisor, DescrCellpaintingSupervisor, DescrSupervisor, GapnetSupervisor
from settings import Settings
import argparse
import os
import torch
from itertools import product

supervisor_dict = {
	'descr': DescrSupervisor,
	'descr_gap': DescrCellpaintingSupervisor,
	'smiles': SmilesSupervisor,
	'smiles_gap': SmilesCellpaintingSupervisor,
	'gapnet': GapnetSupervisor
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

	settings.architecture.model_type = 'gapnet'
	print('Model type: {}'.format(settings.architecture.model_type))

	# done: lrs 0.0001, lrs_gap, 0.0001, 0.001

	lrs = [0.001, 0.01, 0.1]
	lrs_gap = [0.0001, 0.001, 0.01]

	settings.log.log_dir = os.path.join(
		settings.log.log_dir,
		settings.data.label_format,
		settings.architecture.model_type)
	print('Log Dir: {}'.format(settings.log.log_dir))

	for lr, lrg in product(lrs,lrs_gap):
		settings.optimiser.learning_rate = lr
		settings.optimiser.learning_rate_gapnet = lrg

		supervisor_class = supervisor_dict[settings.architecture.model_type]
		supervisor = supervisor_class(settings)
		perfs = supervisor.cross_validate()
