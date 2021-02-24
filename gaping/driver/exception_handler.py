from typing_extensions import Protocol

from abc import abstractmethod


class ExceptionHandler(Protocol):

  @abstractmethod
  def handle_event_exception(self, ex, sequence, event):
    ...

  @abstractmethod
  def handle_on_start_exception(self, ex):
    ...

  @abstractmethod
  def handle_on_shutdown_exception(self, ex):
    ...




