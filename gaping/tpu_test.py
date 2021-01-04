import os
import sys
sys.path += [os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')]

import tensorflow as tf
import tensorflow.compat.v1 as tf1

from absl import flags
from absl.testing import parameterized

from gaping import test_utils


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

if __name__ == "__main__":
  import gaping.wrapper
  with gaping.wrapper.patch_tensorflow():
    tf.test.main()

