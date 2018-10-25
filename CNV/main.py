from supervisors.kfold_superviser import KFoldSupervisor
from net.cnv_net import CNVNet
from collections import OrderedDict
from resources.cellpainting_dataset import CellpaintingDataset
from resources.transforms import ToTensor
from torchvision.transforms import Compose

from settings import Settings

if __name__=='__main__':

	settings_file = 'settings/defaults.json'

	settings = Settings.from_json_file(settings_file)

	supervisor = KFoldSupervisor(settings=settings) #dataset=dataset, model=model)
	supervisor.train(epochs=10)