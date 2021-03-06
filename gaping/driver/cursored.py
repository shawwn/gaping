from typing_extensions import Protocol

from abc import abstractmethod


class Cursored(Protocol):

  @abstractmethod
  def get_cursor(self):
    ...

