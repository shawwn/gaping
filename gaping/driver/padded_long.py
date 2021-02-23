from .. import driver as driver_lib

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import critical_section_ops

def make_mutex(name, shared=False):
  if not shared:
    name = driver_lib.get().uniq(name)
  return critical_section_ops.CriticalSection(name=name, shared_name=name)

from .. import tf_tools as tft

class PaddedLong:
  def __init__(self, initial_value=None, dtype=tf.int64, shape=(), shared=False, name='x'):
    self.var = driver_lib.localvar(name=name, dtype=dtype, shape=shape, shared=shared, initial_value=initial_value)

  def get(self):
    def on_cpu():
      with ops.colocate_with(self.var):
        return self.var.read_value()
    return tft.tpu_cpu(on_cpu)

  def initialized_value(self):
    def on_cpu():
      with ops.colocate_with(self.var):
        return self.var.initialized_value()
    return tft.tpu_cpu(on_cpu)

  def set(self, value, read_value=False, use_locking=True):
    def on_cpu(value):
      with ops.colocate_with(self.var):
        v = tf.cast(value, dtype=self.var.dtype)
        return self.var.assign(value, read_value=read_value, use_locking=use_locking)
    return tft.tpu_cpu(on_cpu, value)

  def increment(self, value=1, read_value=True, use_locking=True):
    def on_cpu(value):
      with ops.colocate_with(self.var):
        v = tf.cast(value, dtype=self.var.dtype)
        return self.var.assign_add(v, read_value=read_value, use_locking=use_locking)
    return tft.tpu_cpu(on_cpu, value)

  @property
  def initializer(self):
    return self.var.initializer
