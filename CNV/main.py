from net.supervisors.kfold_superviser import KFoldSupervisor
from net.supervisors.transfer_learn_supervisor import TransferLearningSupervisor
from net.supervisors.lstm_supervisor import LSTMSupervisor
from settings.settings import Settings
import argparse
import os

if __name__=='__main__':

	os.environ['CUDA_VISIBLE_DEVICES'] = '1'

	argparser = argparse.ArgumentParser(description='Train DILI model')
	argparser.add_argument(r'-s', r'--settings', type=str, help=r'settings file to use')

	args = argparser.parse_args()

	print('Current working directory: {}'.format(os.getcwd()))
	print('Current directory: {}'.format(os.path.dirname(os.path.realpath(__file__))))

	settings_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'settings', args.settings)

	settings = Settings.from_json_file(settings_file)

	#supervisor = KFoldSupervisor(settings=settings)

	#supervisor = TransferLearningSupervisor(settings=settings, model_name='vgg')

	supervisor = LSTMSupervisor(settings=settings)
	supervisor.train()

	print('finished')