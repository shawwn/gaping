import os
import sys
sys.path += [os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')]

import tensorflow as tf
import tensorflow.compat.v1 as tf1

from absl import flags
from absl.testing import parameterized

from gaping import test_utils
from gaping import wrapper

from gaping.wrapper import tpu_ops
from tensorflow.python.tpu import tpu


class TpuTest(parameterized.TestCase, test_utils.GapingTestCase):

  def setUp(self):
    super().setUp()

  def test_001_tpu_session(self):
    with self.session().as_default() as session:
      devices = session.list_devices()
      for device in devices:
        self.log(device)
      self.assertTrue(any([':TPU:' in device.name for device in devices]))

  def test_002_tpu_graph(self):
    with tf.Graph().as_default(), self.session().as_default() as session:
      self.assertNotEqual(session.graph, self.cached_session().graph)

  def test_003_add_tpu_cpu(self):
    with tf.Graph().as_default():
      self.assertEqual(3, self.evaluate(tf.add(1, 2)))

  def tpu_core_count(self):
    return 8 # TODO

  def test_003_add_tpu_cores(self):
    with tf.Graph().as_default():
      self.assertAllEqual([[3., 3., 3., 3., 3., 3., 3., 3.]],
          self.evaluate(wrapper.tpu_shard(lambda: tf.add(1, 2))))

  def test_004_permute(self):
    with tf.Graph().as_default():
      inputs = [[tf.constant([x+1], dtype=tf.float32) for x in range(8)]]
      def on_cores(tensor):
        return tpu.tpu_ops.collective_permute(tensor,
            [[0, 1], [1, 0], [2, 3], [3, 2], [4, 5], [5, 4], [6, 7], [7, 6]])
      self.assertAllEqual([[[2.], [1.], [4.], [3.], [6.], [5.], [8.], [7.]]],
          self.evaluate(wrapper.tpu_shard(on_cores, inputs=inputs)))


if __name__ == "__main__":
  import gaping.wrapper
  with gaping.wrapper.patch_tensorflow():
    tf.test.main()

