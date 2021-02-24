from typing_extensions import Protocol

from abc import abstractmethod


class Runnable(Protocol):

  @abstractmethod
  def run(self):
    ...


