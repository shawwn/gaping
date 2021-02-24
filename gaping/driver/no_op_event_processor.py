from typing_extensions import Protocol

from abc import abstractmethod

from . import event_processor
from . import initializable
from . import atomic_boolean
from . import sequence
from . import sequencer as sequencer_lib
from . import util


class NoOpEventProcessor(event_processor.EventProcessor, initializable.Initializable):

  def __init__(self, sequencer):
    self.sequence = SequencerFollowingSequence(sequencer)
    self.running = atomic_boolean.AtomicBoolean(False)

  def get_initializers(self):
    return super().get_initializers() + \
        self.running.get_initializers() + \
        self.sequence.get_initializers()

  def get_sequence(self):
    return self.sequence

  def halt(self):
    return self.running.set(False)

  def is_running(self):
    return self.running.get()

  def run(self):
    ok = self.running.compare_and_set(False, True)
    return util.check(ok, ok, "Thread is already running")


class SequencerFollowingSequence(sequence.Sequence):
  def __init__(self, sequencer):
    super().__init__(sequencer_lib.Sequencer.INITIAL_CURSOR_VALUE)
    self.sequencer = sequencer

  def get(self):
    return self.sequencer.get_cursor()
