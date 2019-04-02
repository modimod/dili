from net.supervisors import DescrCellpaintingMultiSupervisor
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

	settings.architecture.model_type = 'descr_gap'
	print('Model type: {}'.format(settings.architecture.model_type))


	# best 3 descr hyperparams after hyperparams tuning
	# for binary
	params = [
		(0.0001, [512, 256, 128, 64], 0.0),
		(0.001, [256, 128], 0.0),
		(0.001, [512, 256, 128], 0.0)]

	# best 3 gapnet hyperparams after hyperparams tuning
	# lrs, lrs_gapnet
	# for binary
	params_g = [
		(0.001, 0.01),
		(0.0001, 0.0001),
		(0.001, 0.0001)]

	# for classification
	# params = [
	# 	(0.0001, [256, 128, 64, 32], 0.0),
	# 	(0.0005, [512, 256, 128, 64], 0.0),
	# 	(0.0001, [512, 256], 0.0)]
	#

	# for ranked
	#params = [
	#	(0.0005, [128, 64, 32], 0.0),
	#	(0.0005, [512, 256, 128, 64], 0.0),
	#	(0.001, [128, 64], 0.0)]

	# lr_gap = [1e-4]
	# fes = [True, *lr_gap]

	settings.architecture.feature_extract = True

	gapnet_image_mode = ['mean', 'all']

	settings.log.log_dir = os.path.join(
		settings.log.log_dir,
		settings.data.label_format,
		settings.architecture.model_type)
	print('Log Dir: {}'.format(settings.log.log_dir))

	# for p, fe, gim in product(params, fes, gapnet_image_mode):
	for p, gim in product(params, gapnet_image_mode):
		settings.optimiser.learning_rate = p[0]
		settings.architecture.fc_hidden_dims = p[1]
		settings.architecture.fc_dropout = p[2]

		settings.architecture.gapnet_image_mode = gim

		# if fe is True:
		# 	settings.architecture.feature_extract = fe
		# else:
		# 	settings.architecture.feature_extract = False
		# 	settings.optimiser.learning_rate_gapnet = fe

		supervisor = DescrCellpaintingMultiSupervisor(settings)
		perfs = supervisor.cross_validate()
