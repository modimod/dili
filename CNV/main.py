from net.supervisors import SmilesCellpaintingSupervisor, SmilesSupervisor, DescrCellpaintingSupervisor, DescrSupervisor, GapnetSupervisor, DescrCellpaintingMultiSupervisor
from settings import Settings
import argparse
import os
import torch
from resources.cellpainting_dataset import CellpaintingDataset
from torch.utils.data import DataLoader

supervisor_dict = {
	'descr': DescrSupervisor,
	'descr_gap': DescrCellpaintingSupervisor,
	'descr_gap_multi': DescrCellpaintingMultiSupervisor,
	'smiles': SmilesSupervisor,
	'smiles_gap': SmilesCellpaintingSupervisor,
	'gapnet': GapnetSupervisor
}

if __name__=='__main__':


	torch.backends.cudnn.benchmark = True

	argparser = argparse.ArgumentParser(description='Train DILI model')
	arg_sub_parsers = argparser.add_subparsers()
	arg_sub_parsers.required = True
	arg_sub_parsers.dest = r'mode'

	train_parser = arg_sub_parsers.add_parser(name=r'train', help=r'train network using specified settings file')
	train_parser.add_argument(r'-s', r'--settings', type=str, help=r'settings file to use', required=True)
	train_parser.add_argument(r'-g', r'--gpus', type=str, help=r'GPU(s) to use, default empty (cpu)', required=False,
						   default='')

	predict_parser = arg_sub_parsers.add_parser(name=r'evaluate', help=r'predict dili outcome of test data on already trained model')
	predict_parser.add_argument(r'-c', r'--checkpoint', type=str, help=r'checkpoint file (pt) to use', required=True)
	predict_parser.add_argument(r'-s', r'--settings', type=str, help=r'settings file to use', required=True)

	args = argparser.parse_args()

	print('GPUs: {}'.format(args.gpus))
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

	# get settings
	settings_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'settings', args.settings)
	settings = Settings.from_json_file(settings_file)

	settings.run.device = 'cuda' if torch.cuda.is_available() else 'cpu'

	settings.architecture.model_type = 'descr'

	if args.mode == r'train':
		print('Used device: {}'.format(settings.run.device))
		print('Feature Extract: {}'.format(settings.architecture.feature_extract))

		supervisor_class = None
		try:
			supervisor_class = supervisor_dict[settings.architecture.model_type]
		except KeyError:
			print('Wrong model type defined in settings')
			exit(-1)

		print('Model type: {}'.format(supervisor_class.__name__))


		supervisor = supervisor_class(settings)
		perfs = supervisor.cross_validate()

		print('best performances')
		for p in perfs:
			print(p.to_dictionary())

	elif args.mode == r'evaluate':

		# load model
		tagger = torch.load(f=args.checkpoint, map_location=r'cpu')
		# tagger = TransferLearningTagger(settings, model_name='gapnet', feature_extract=False)
		tagger.device = r'cuda' if torch.cuda.is_available() and settings.run.cuda else r'cpu'

		# get dataset
		testset = CellpaintingDataset(csv_file=settings.data.csv_file_test, root_dir=settings.data.root_dir, file_ext=settings.data.file_ext, mode_test=settings.run.test_mode, eval=True)
		testloader = DataLoader(dataset=testset, batch_size=settings.run.batch_size_eval)

		# run eval
		performance = tagger.evaluate(dataloader=testloader, eval_col=settings.data.eval_col)

		print(performance.acc)

	else:
		raise ValueError(r'Invalid mode specified! Aborting...')
