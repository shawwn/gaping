from .. import driver as driver_lib

import tensorflow as tf

from . import padded_long
from . import sequence
from . import util
from . import sequencer

class AbstractSequencer(sequencer.Sequencer):
  def __init__(self, buffer_size, wait_strategy):
    super().__init__()
    self.cursor = sequence.Sequence(sequencer.Sequencer.INITIAL_CURSOR_VALUE)
    self.gating_sequences = set()
    with util.dep(tf.assert_greater_equal(buffer_size, 1)):
      self.buffer_size = buffer_size
      self.wait_strategy = wait_strategy

  def get_initializers(self):
    return [self.cursor.initializer]

  @property
  def initializer(self):
    return tf.group(self.get_initializers())

  def get_cursor(self):
    return self.cursor.get()

  def get_buffer_size(self):
    return self.buffer_size

  def add_gating_sequences(self, *gating_sequences):
    for seq in gating_sequences:
      self.gating_sequences.add(seq)

  def remove_gating_sequence(self, seq):
    self.gating_sequences.remove(seq)

  def get_minimum_sequence(self):
    return util.get_minimum_sequence(self.gating_sequences, self.cursor.get())

  def signal_all_when_blocking(self, op):
    if self.wait_strategy is None:
      return op
    with util.dep(op):
      with util.dep(self.wait_strategy.signal_all_when_blocking()):
        return tf.identity(op)

  def new_barrier(self, *sequences_to_track):
    return ProcessingSequenceBarrier(self, self.wait_strategy, self.cursor, sequences_to_track)

  def new_poller(self, data_provider, *gating_sequences):
    return EventPoller.new_instance(data_provider, self, sequence.Sequence(), self.cursor, self.gating_sequences)

  def __str__(self):
    return type(self).__name__ + "{" + \
        "waitStrategy=" + str(self.wait_strategy) + \
        ", cursor=" + str(self.cursor) + \
        ", gatingSequences=[" + ', '.join([str(x) for x in self.gating_sequences]) + "]" + \
        "}"

  def __repr__(self):
    return str(self)




class SingleProducerSequencer(AbstractSequencer):
  def __init__(self, buffer_size, wait_strategy):
    super().__init__(buffer_size=buffer_size, wait_strategy=wait_strategy)
    self.next_value = padded_long.PaddedLong(sequence.Sequence.INITIAL_VALUE)
    self.cached_value = padded_long.PaddedLong(sequence.Sequence.INITIAL_VALUE)

  def get_initializers(self):
    return super().get_initializers() + [
      self.next_value.initializer,
      self.cached_value.initializer,
      ];

  def get_wrap_point(self, required_capacity):
    next_value = self.next_value.get()
    return (next_value + required_capacity) - self.buffer_size

  def has_available_capacity(self, required_capacity):
    return self._has_available_capacity(required_capacity, do_store=False)

  def _has_available_capacity(self, required_capacity, do_store):

    def yes():
      def store():
        if do_store:
          return self.cursor.set_volatile(self.next_value.get())
        else:
          return tf.no_op()
      with util.dep(store()):
        min_sequence = util.get_minimum_sequence(self.gating_sequences, self.next_value.get())
        with util.dep(self.cached_value.set(min_sequence)):
          return self.get_wrap_point(required_capacity) <= min_sequence
    return tf.cond(
        tf.logical_or(
          (self.get_wrap_point(required_capacity) > self.cached_value.get()),
          (self.cached_value.get() > self.next_value.get()),
          ),
        yes,
        lambda: True)

  def try_next_n(self, n):
    n = util.ti64(n)
    n = util.check(n, n >= 1, "n must be > 0") 
    if False:
      n = util.check(n, self._has_available_capacity(n, True), "insufficient capacity")
      next_sequence = self.next_value.increment(n)
      return next_sequence
    else:
      return tf.cond(self._has_available_capacity(n, True),
          lambda: self.next_value.increment(n),
          lambda: util.ti64(-1))

  def next_n(self, n):
    raise NotImplementedError()

  def remaining_capacity(self):
    next_value = self.next_value.get()
    consumed = util.get_minimum_sequence(self.gating_sequences, next_value)
    produced = next_value
    return self.get_buffer_size() - (produced - consumed)

  def claim(self, sequence):
    return self.next_value.set(sequence)

  def publish(self, sequence):
    op = self.cursor.set(sequence)
    op = self.signal_all_when_blocking(op)
    return op

  def publish_range(self, lo, hi):
    return self.publish(hi)

  def is_available(self, sequence):
    return sequence <= self.cursor.get()

  def get_highest_published_sequence(self, lower_bound, available_sequence):
    return available_sequence



class MultiProducerSequencer(AbstractSequencer):
  def __init__(self, buffer_size, wait_strategy):
    super().__init__(buffer_size=buffer_size, wait_strategy=wait_strategy)
    self.gating_sequence_cache = padded_long.PaddedLong(sequencer.Sequencer.INITIAL_CURSOR_VALUE)

  def get_initializers(self):
    return super().get_initializers() + [
      self.gating_sequence_cache.initializer,
      ];

  def has_available_capacity(self, required_capacity):
    return self._has_available_capacity(self.gating_sequences, required_capacity, self.cursor.get())

  def _has_available_capacity(self, gating_sequences, required_capacity, cursor_value):
    wrap_point = (cursor_value + required_capacity) - self.buffer_size
    cached_gating_sequence = self.gating_sequence_cache.get()
    def yes():
      min_sequence = util.get_minimum_sequence(gating_sequences, cursor_value)
      with util.dep(self.gating_sequence_cache.set(min_sequence)):
        return wrap_point <= min_sequence
    return tf.cond(
        tf.logical_or(
          (wrap_point > cached_gating_sequence),
          (cached_gating_sequence > cursor_value),
          ),
        yes,
        lambda: True)

  def claim(self, sequence):
    return self.cursor.set(sequence)

  def try_next_n(self, n):
    n = util.ti64(n)
    n = util.check(n, n >= 1, "n must be > 0") 
    def body(i):
      current = self.cursor.get()
      next = current + n
      with util.dep(util.check(n, self._has_available_capacity(self.gating_sequences, n, current), "insufficient capacity")):
        return tf.equal(False, self.cursor.compare_and_set(current, next))
    i = tf.while_loop(body, lambda i: i + 1, [0], parallel_iterations=1, maximum_iterations=1000)
    with util.dep(tf.assert_less(i, 1000)):
      return self.cursor.get()

  def remaining_capacity(self):
    consumed = util.get_minimum_sequence(self.gating_sequences, self.cursor.get())
    produced = self.cursor.get()
    return self.get_buffer_size() - (produced - consumed)


