import copy, json, os
from utils.decorators import method_accepts, method_bounds


class DataSettings(object):
	"""
	Class holding all data specific settings.
	"""

	def __init__(self, settings: dict) -> None:
		"""
		Initialise all data specific settings.

		:param settings: settings dictionary to initialise from
		"""

		self.csv_file = settings[r'csv_file']
		self.root_dir = settings[r'root_dir']
		self.file_ext = settings[r'file_ext']
		self.smiles_csv_file = settings[r'smiles_csv_file']

		self.training_file_path = settings[r'training_file_path']
		self.training_sample_separator = settings[r'training_sample_separator']
		self.training_distinct_samples = settings[r'training_distinct_samples']
		self.evaluation_file_path = settings[r'evaluation_file_path']
		self.evaluation_sample_separator = settings[r'evaluation_sample_separator']
		self.evaluation_distinct_samples = settings[r'evaluation_distinct_samples']


class LogSettings(object):
	"""
	Class holding all log specific settings.
	"""

	def __init__(self, settings: dict) -> None:
		"""
		Initialise all log specific settings.

		:param settings: settings dictionary to initialise from
		"""
		self.checkpoint_directory = settings[r'checkpoint_directory']
		self.summary_directory = settings[r'summary_directory']
		self.overwrite = settings[r'overwrite']


class RunSettings(object):
	"""
	Class holding all run specific settings.
	"""

	def __init__(self, settings: dict) -> None:
		"""
		Initialise all run specific settings.

		:param settings: settings dictionary to initialise from
		"""
		self.seed = settings[r'seed']
		self.cuda = settings[r'cuda']
		self.epochs = settings[r'epochs']
		self.early_stop = settings[r'early_stop']
		self.evaluation_interval = settings[r'evaluation_interval']
		self.batch_size = settings[r'batch_size']
		self.batch_size_eval = settings[r'batch_size_eval']
		self.shuffle = settings[r'shuffle']
		self.test_mode = settings[r'test_mode']
		self.weight_classes = settings[r'weight_classes']
		self.weight_classes_normalised = settings[r'weight_classes_normalised']
		self.weight_classes_deleted = settings[r'weight_classes_deleted']


class ArchitectureSettings(object):
	"""
	Class holding all architecture specific settings.
	"""

	def __init__(self, settings: dict) -> None:
		"""
		Initialise all architecture specific settings.

		:param settings: settings dictionary to initialise from
		"""
		self.input_dropout = settings[r'input_dropout']
		#self.fc_layers = settings[r'fc_layers']
		self.fc_unit_1 = settings[r'fc_unit_1']
		self.fc_unit_2 = settings[r'fc_unit_2']

		self.lstm_hidden_dim = settings[r'lstm_hidden_dim']
		self.lstm_num_layers = settings[r'lstm_num_layers']
		self.lstm_dropout = settings[r'lstm_dropout']
		self.lstm_sliding_window = settings[r'lstm_sliding_window']


class OptimiserSettings(object):
	"""
	Class holding all optimiser specific settings.
	"""

	def __init__(self, settings: dict) -> None:
		"""
		Initialise all optimiser specific settings.

		:param settings: settings dictionary to initialise from
		"""
		self.learning_rate = settings[r'learning_rate']


class PreProcessingSettings(object):
	"""
	Class holding all pre-processing specific settings.
	"""

	def __init__(self, settings: dict) -> None:
		"""
		Initialise all pre-processing specific settings.

		:param settings: settings dictionary to initialise from
		"""
		self.window_offset = settings[r'window_offset']


class Settings(object):
	"""
	Class holding all settings for reCOIL.
	"""

	def __init__(self, settings: dict) -> None:
		"""
		Initialise all settings.

		:param settings: settings dictionary to initialise from
		"""
		defaults = os.path.join(os.path.dirname(os.path.realpath(__file__)), r'defaults.json')
		with open(file=defaults, mode=r'r') as defaults_json:
			defaults = json.load(fp=defaults_json)

		def merge_settings(_defaults: dict, _custom: dict) -> dict:
			_current = copy.deepcopy(_defaults)
			for settings_type, settings_value in _custom.items():
				if settings_type in _defaults:
					_current[settings_type].update(settings_value)
				else:
					_current[settings_type] = settings_value

			return _current

		settings = merge_settings(_defaults=defaults, _custom=settings)
		self.pre_processing = settings[r'pre_processing']
		self.optimiser = settings[r'optimiser']
		self.architecture = settings[r'architecture']
		self.run = settings[r'run']
		self.data = settings[r'data']
		self.log = settings[r'log']

	@classmethod
	def from_json_file(cls, file: str) -> r'Settings':
		with open(file=file, mode=r'r') as custom:
			return cls(json.load(fp=custom))

	@property
	def pre_processing (self) -> PreProcessingSettings:
		return self._pre_processing

	@pre_processing.setter
	@method_accepts(PreProcessingSettings)
	def pre_processing (self, pre_processing_new: dict) -> None:
		self._pre_processing = PreProcessingSettings(settings=pre_processing_new)

	@property
	def optimiser (self) -> OptimiserSettings:
		return self._optimiser

	@optimiser.setter
	@method_accepts(OptimiserSettings)
	def optimiser (self, optimiser_new: dict) -> None:
		self._optimiser = OptimiserSettings(settings=optimiser_new)

	@property
	def architecture (self) -> ArchitectureSettings:
		return self._architecture

	@architecture.setter
	@method_accepts(ArchitectureSettings)
	def architecture (self, architecture_new: dict) -> None:
		self._architecture = ArchitectureSettings(settings=architecture_new)

	@property
	def run (self) -> RunSettings:
		return self._run

	@run.setter
	@method_accepts(RunSettings)
	def run (self, run_new: dict) -> None:
		self._run = RunSettings(settings=run_new)

	@property
	def data (self) -> DataSettings:
		return self._data

	@data.setter
	@method_accepts(DataSettings)
	def data (self, data_new: dict) -> None:
		self._data = DataSettings(settings=data_new)

	@property
	def log (self) -> LogSettings:
		return self._log

	@log.setter
	@method_accepts(LogSettings)
	def log (self, log_new: dict) -> None:
		self._log = LogSettings(settings=log_new)


