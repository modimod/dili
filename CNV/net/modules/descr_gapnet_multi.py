import torch.nn as nn
import torch
from torch.nn import BCEWithLogitsLoss

from net.modules.MultiOuts import MultiOuts, MultiOutsBinary
from net.modules.gapnet import GAPNet02
from net.modules.base_modules import FCModule, FCDynamicModule
from utils.constants import pandas_cols, tasks, tasks_rank, tasks_idx, descr_dim
from net.modules.descr_gapnet import DescrGapnetModule

class DescrGapnetMultiModule(DescrGapnetModule):

	def forward (self, descr, images):
		if self.settings.architecture.gapnet_image_mode == 'mean':

			descr_out = self.descr(descr)

			# get all outputs of the list of images
			gapnet_outs = list()

			for im in images:

				out = self.gapnet(im)		# output of all images in that list entry
				out = out.mean(dim=0)		# take mean of outputs

				gapnet_outs.append(out)

			# stack batch outputs
			gapnet_out = torch.stack(gapnet_outs)

			x = torch.cat((descr_out, gapnet_out), dim=1)

		else: # don't take mean of images but all
			descr_out = self.descr(descr)

			# get all outputs of the list of images
			outs = list()

			for d,im in zip(descr_out,images):
				out = self.gapnet(im)  				# output of all images in that list entry
				d_ext = d.repeat(len(im), 1)		# repeat descr_out as often as count of images for that image

				concat = torch.cat((d_ext, out), dim=1)

				outs.append(concat)

			# stack batch outputs
			x = torch.stack(outs)

		return self.multiout(x)


class DescrGapnetMultiBinaryModule(DescrGapnetMultiModule):
	def __init__(self, settings, feature_extract=True):
		super().__init__(settings, feature_extract)

		self.multiout = MultiOutsBinary(self.multiout_in, len(tasks))
		self.loss = self.multiout.loss


class DescrGapnetMultiRankedModule(DescrGapnetMultiModule):
	def __init__(self, settings, feature_extract=True):
		super().__init__(settings, feature_extract)

		self.multiout = MultiOutsBinary(self.multiout_in, sum(tasks_rank.values()))
		self.loss = self.multiout.loss

