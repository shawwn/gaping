from .. import driver as driver_lib
from .. import tf_tools as tft

import tensorflow as tf

from . import padded_long
from . import sequence
from . import util
from . import pipe


class CountDownLatch:
  def __init__(self, n, shared_name=None):
    self.n = util.ti64(n)
    builder = pipe.PipeBuilder()
    builder.input('value', tf.int64)
    self.stager = builder.get(shared_name=shared_name)

  def _size(self):
    return util.ti64(self.stager.size())

  def wait(self, timeout=None, interval=10/1000):
    if timeout is None:
      with util.dep(tf.group([self.stager.peek(self.n - 1)])):
        return tf.constant(True)
    else:
      def running(elapsed):
        return tf.logical_and(elapsed < timeout, self.get_count() > 0)
      def result(elapsed):
        return self.get_count() <= 0
      return util.check_every(interval, running, result)

  def count_down(self):
    with util.dep(self.stager.put(tf.minimum(self.n - 1, self._size()), {'value': tft.tpu_nanos()})):
      return util.ti64(self.get_count())

  def get_count(self):
    return self.n - self._size()
