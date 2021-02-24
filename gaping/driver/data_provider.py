from typing_extensions import Protocol

from abc import abstractmethod

class DataProvider(Protocol):

  @abstractmethod
  def get(self, sequence):
    ...

