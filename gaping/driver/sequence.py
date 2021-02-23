from .. import driver as driver_lib

import tensorflow as tf

from . import padded_long

class Value:
  def __init__(self, *args, **kws):
    self.value = padded_long.PaddedLong(*args, **kws)

  @property
  def initializer(self):
    return self.value.initializer


class Sequence(Value):

  INITIAL_VALUE = -1

  def __init__(self, initial_value=None, name='x', shared=False):
    if initial_value is None:
      initial_value = Sequence.INITIAL_VALUE
    super().__init__(initial_value, dtype=tf.int64, name=name, shared=shared)
    self._get_op = self.value.initialized_value()

  def get(self):
    return self.value.get()

  def set(self, value):
    return self.value.set(value)

  def set_volatile(self, value):
    return self.value.set(value)

  def compare_and_set(self, expected_value, new_value):
    def yes():
      with tf.control_dependencies([self.value.set(new_value)]):
        return tf.constant(True)
    def no():
      return tf.constant(False)
    return tf.cond(tf.equal(self.get(), expected_value), yes, no)

  def increment_and_get(self):
    return self.add_and_get(1)

  def add_and_get(self, increment):
    return self.value.increment(increment)

  def __str__(self):
    return str(self._get_op.eval())

  def __repr__(self):
    return 'Sequence({})'.format(str(self))
