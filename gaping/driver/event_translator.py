from typing_extensions import Protocol

from abc import abstractmethod

class EventTranslator(Protocol):

  @abstractmethod
  def translate_to(self, event, sequence, *args, **kws):
    ...

  def __call__(self, event, sequence, *args, **kws):
    return self.translate_to(event, sequence, *args, **kws)


