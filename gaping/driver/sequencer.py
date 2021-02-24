from typing_extensions import Protocol

from abc import abstractmethod

from . import cursored
from . import sequenced

class Sequencer(cursored.Cursored, sequenced.Sequenced):
  INITIAL_CURSOR_VALUE = -1

  @abstractmethod
  def claim(self, sequence):
    ...

  @abstractmethod
  def is_available(self, sequence):
    ...

  @abstractmethod
  def add_gating_sequences(self, *gating_sequences):
    ...

  @abstractmethod
  def remove_gating_sequence(self, sequence):
    ...

  @abstractmethod
  def new_barrier(self, *sequences_to_track):
    ...

  @abstractmethod
  def get_minimum_sequence(self):
    ...

  @abstractmethod
  def get_highest_published_sequence(self, next_sequence, available_sequence):
    ...

  #@abstractmethod
  def new_poller(self, provider, *gating_sequences):
    ...

