from torchvision import models, transforms
import torch.nn as nn
from net.modules.MultiOuts import MultiOuts
from net.modules.gapnet import GAPNet02

from resources.transforms import RandomCrop,Rescale,ToTensor,Normalize, TakeChannels

import os
import torch

def initialize_model (settings, model_name, num_classes, feature_extract, use_pretrained=True):
	# Initialize these variables which will be set in this if statement. Each of these
	#   variables is model specific.
	model_ft = None
	input_size = 0

	if model_name == "resnet":
		""" Resnet18
		"""
		model_ft = models.resnet18(pretrained=use_pretrained)
		set_parameter_requires_grad(model_ft, feature_extract)
		num_ftrs = model_ft.fc.in_features

		multiout = MultiOuts(num_ftrs)
		model_ft.fc = multiout
		model_ft.loss = multiout.masked_loss

		# model_ft.fc = nn.Linear(num_ftrs, num_classes)
		input_size = 224

	elif model_name == "alexnet":
		""" Alexnet
		"""
		model_ft = models.alexnet(pretrained=use_pretrained)
		set_parameter_requires_grad(model_ft, feature_extract)
		num_ftrs = model_ft.classifier[6].in_features

		multiout = MultiOuts(num_ftrs)
		model_ft.classifier[6] = multiout
		model_ft.loss = multiout.masked_loss

		# model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
		input_size = 224

	elif model_name == "vgg":
		""" VGG11_bn
		"""
		model_ft = models.vgg11_bn(pretrained=use_pretrained)
		set_parameter_requires_grad(model_ft, feature_extract)
		num_ftrs = model_ft.classifier[6].in_features

		multiout = MultiOuts(num_ftrs)
		model_ft.classifier[6] = multiout
		model_ft.loss = multiout.masked_loss
		#model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
		input_size = 224

	elif model_name == "gapnet":
		model_ft = GAPNet02(input_shape=(5, 520, 696), fc_units=1024, dropout=0, num_classes=209)

		device = ':'.join([r'cuda',str(settings.run.gpu)]) if torch.cuda.is_available() and settings.run.cuda else r'cpu'
		checkpoint = torch.load(f=settings.data.pretrained_gapnet, map_location=device)

		# rename checkpoint state_dict names
		new_state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}

		model_ft.load_state_dict(new_state_dict)

		set_parameter_requires_grad(model_ft, feature_extract)

		num_ftrs = model_ft.classifier[3].out_features

		multiout = MultiOuts(num_ftrs)
		model_ft.classifier[6] = multiout
		model_ft.loss = multiout.masked_loss

		input_size = (1, 1)


	elif model_name == "squeezenet":
		""" Squeezenet
		"""
		model_ft = models.squeezenet1_0(pretrained=use_pretrained)
		set_parameter_requires_grad(model_ft, feature_extract)
		model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
		model_ft.num_classes = num_classes
		input_size = 224

	elif model_name == "densenet":
		""" Densenet
		"""
		model_ft = models.densenet121(pretrained=use_pretrained)
		set_parameter_requires_grad(model_ft, feature_extract)
		num_ftrs = model_ft.classifier.in_features

		multiout = MultiOuts(num_ftrs)
		model_ft.classifier = multiout
		model_ft.loss = multiout.masked_loss

		# model_ft.classifier = nn.Linear(num_ftrs, num_classes)
		input_size = 224

	elif model_name == "inception":
		""" Inception v3 
		Be careful, expects (299,299) sized images and has auxiliary output
		"""
		model_ft = models.inception_v3(pretrained=use_pretrained)
		set_parameter_requires_grad(model_ft, feature_extract)
		# Handle the auxilary net
		num_ftrs = model_ft.AuxLogits.fc.in_features

		multiout = MultiOuts(num_ftrs)
		model_ft.AuxLogits.fc = multiout

		#model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
		# Handle the primary net
		num_ftrs = model_ft.fc.in_features

		multiout = MultiOuts(num_ftrs)
		model_ft.fc = multiout

		model_ft.loss = multiout.masked_loss

		#model_ft.fc = nn.Linear(num_ftrs, num_classes)
		input_size = 299

	else:
		print("Invalid model name, exiting...")
		exit()

	return model_ft, input_size


def set_parameter_requires_grad (model, feature_extracting):
	if feature_extracting:
		for param in model.parameters():
			param.requires_grad = False


def get_data_transforms(input_size):
	data_transforms = {
		'train': transforms.Compose([
			TakeChannels(),
			RandomCrop(input_size),
			transforms.RandomHorizontalFlip(),
			ToTensor(),
			Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
		'val': transforms.Compose([
			TakeChannels(),
			Rescale((input_size,input_size)),
			ToTensor(),
			Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
	}

	return data_transforms