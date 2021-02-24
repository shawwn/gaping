from typing_extensions import Protocol

from abc import abstractmethod

class Sequenced(Protocol):

  @abstractmethod
  def get_buffer_size(self):
    ...

  @abstractmethod
  def has_available_capacity(self, required_capacity):
    ...

  @abstractmethod
  def remaining_capacity(self):
    ...

  def next(self):
    return self.next_n(1)

  @abstractmethod
  def next_n(self, n):
    ...

  def try_next(self):
    return self.try_next_n(1)

  @abstractmethod
  def try_next_n(self, n):
    ...

  @abstractmethod
  def publish(self, sequence):
    ...

  @abstractmethod
  def publish_range(self, lo, hi):
    ...
