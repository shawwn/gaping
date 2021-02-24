from typing_extensions import Protocol

from abc import abstractmethod

class EventTranslator(Protocol):

  @abstractmethod
  def translate_to(self, event, sequence):
    ...



