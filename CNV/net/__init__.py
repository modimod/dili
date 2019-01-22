import warnings

from abc import ABCMeta, abstractmethod
from tqdm import TqdmSynchronisationWarning
from typing import Any

# Ignore tqdm synchronisation warning.
warnings.filterwarnings(action=r'ignore', category=TqdmSynchronisationWarning)


class BaseTagger(metaclass=ABCMeta):
    """
    Class representing the base network used in all taggers.
    """

    @abstractmethod
    def fit(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def predict(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def reset(self, *args, **kwargs) -> Any:
        pass


class BaseSupervisor(metaclass=ABCMeta):
    """
    Class representing the base supervisor used for operating networks.
    """

    @abstractmethod
    def fit(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def reset(self, *args, **kwargs) -> Any:
        pass
