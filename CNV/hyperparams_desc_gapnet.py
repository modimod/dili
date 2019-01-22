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

	settings.architecture.model_type = 'descr_gap'
	print('Model type: {}'.format(settings.architecture.model_type))


	# for binary
	# params = [
	# 	(0.0001, [512, 256, 128, 64], 0.0),
	# 	(0.001, [256, 128], 0.0),
	# 	(0.001, [512, 256, 128], 0.0)]

	# for classification
	# params = [
	# 	(0.0001, [256, 128, 64, 32], 0.0),
	# 	(0.0005, [512, 256, 128, 64], 0.0),
	# 	(0.0001, [512, 256], 0.0)]
	#

	# for ranked
	params = [
		(0.0005, [128, 64, 32], 0.0),
		(0.0005, [512, 256, 128, 64], 0.0),
		(0.001, [128, 64], 0.0)]

	lr_gap = [1e-5, 1e-5*5, 1e-4, 1e-4*5]
	fes = [True, *lr_gap]

	settings.log.log_dir = os.path.join(
		settings.log.log_dir,
		settings.data.label_format,
		settings.architecture.model_type)
	print('Log Dir: {}'.format(settings.log.log_dir))

	for p, fe in product(params, fes):
		settings.optimiser.learning_rate = p[0]
		settings.architecture.fc_hidden_dims = p[1]
		settings.architecture.fc_dropout = p[2]

		if fe is True:
			settings.architecture.feature_extract = fe
		else:
			settings.architecture.feature_extract = False
			settings.optimiser.learning_rate_gapnet = fe

		supervisor_class = supervisor_dict[settings.architecture.model_type]
		supervisor = supervisor_class(settings)
		perfs = supervisor.cross_validate()
