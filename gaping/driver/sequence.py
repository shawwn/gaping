from .. import driver as driver_lib

import tensorflow as tf

from . import padded_long
from . import initializable


class Value(initializable.Initializable):
  def __init__(self, initial_value, dtype, shape=(), name='value', shared=False):
    self.value = padded_long.PaddedLong(
        initial_value=initial_value, dtype=dtype, shape=shape, name=name, shared=shared)
    self._get_op = self.value.initialized_value()

  def get_initializers(self):
    return super().get_initializers() + \
        self.value.get_initializers()

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

  def __str__(self):
    return str(self._get_op.eval())

  def __repr__(self):
    return type(self).__name__ + '({})'.format(str(self))


class Sequence(Value):

  INITIAL_VALUE = -1

  def __init__(self, initial_value=None, name='sequence', shared=False):
    if initial_value is None:
      initial_value = Sequence.INITIAL_VALUE
    super().__init__(initial_value, dtype=tf.int64, name=name, shared=shared)

  def increment_and_get(self):
    return self.add_and_get(1)

  def add_and_get(self, increment):
    return self.value.increment(increment)
