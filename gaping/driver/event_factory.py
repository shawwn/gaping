from typing_extensions import Protocol

from abc import abstractmethod


class EventFactory(Protocol):

  @abstractmethod
  def new_instance(self, buffer_size=None):
    ...


