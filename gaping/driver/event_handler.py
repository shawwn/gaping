from typing_extensions import Protocol

from abc import abstractmethod


class EventHandler(Protocol):
  @abstractmethod
  def on_event(self, event, sequence, end_of_batch):
    ...


class LatchEventHandler(EventHandler):
  def __init__(self, latch):
    self.latch = latch

  def on_event(self, event, sequence, end_of_batch):
    return self.latch.count_down()
  


