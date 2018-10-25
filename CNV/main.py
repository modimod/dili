from supervisors.kfold_superviser import KFoldSupervisor
from settings import Settings

if __name__=='__main__':

	settings_file = 'settings/local.json'

	settings = Settings.from_json_file(settings_file)

	supervisor = KFoldSupervisor(settings=settings) #dataset=dataset, model=model)
	supervisor.train(epochs=10)