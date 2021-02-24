from typing_extensions import Protocol

from abc import abstractmethod

from . import runnable


class EventProcessor(runnable.Runnable):

  @abstractmethod
  def get_sequence(self):
    ...

  @abstractmethod
  def halt(self):
    ...

  @abstractmethod
  def is_running(self):
    ...


