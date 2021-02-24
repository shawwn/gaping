from typing_extensions import Protocol

from abc import abstractmethod

class EventSink(Protocol):

  @abstractmethod
  def publish_event(self, translator, *args):
    ...

  @abstractmethod
  def try_publish_event(self, translator, *args):
    ...


