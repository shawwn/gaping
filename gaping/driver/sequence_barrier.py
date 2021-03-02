from typing_extensions import Protocol

from abc import abstractmethod


class SequenceBarrier(Protocol):

  @abstractmethod
  def get_cursor(self):
    ...

  @abstractmethod
  def is_alerted(self):
    ...

  @abstractmethod
  def alert(self):
    ...

  @abstractmethod
  def clear_alert(self):
    ...

  @abstractmethod
  def check_alert(self):
    ...


