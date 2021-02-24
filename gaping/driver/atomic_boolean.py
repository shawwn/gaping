from . import sequence

import tensorflow as tf

class AtomicBoolean(sequence.Value):
  def __init__(self, initial_value=False, name='flag', shared=False):
    super().__init__(initial_value=initial_value, dtype=tf.bool, name=name, shared=shared)
