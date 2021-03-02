from typing_extensions import Protocol

from abc import abstractmethod


class WaitStrategy(Protocol):

  @abstractmethod
  def wait_for(self, sequence, cursor, dependent_sequence, barrier):
    ...

  @abstractmethod
  def signal_all_when_blocking(self):
    ...

