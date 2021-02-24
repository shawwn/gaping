from typing_extensions import Protocol

from abc import abstractmethod


class LifecycleAware(Protocol):

  @abstractmethod
  def on_start(self):
    ...

  @abstractmethod
  def on_shutdown(self):
    ...


