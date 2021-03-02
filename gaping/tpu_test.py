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

def requeue(queue):
  out = queue.dequeue()
  vs = [tf.Variable(x, use_resource=True, trainable=False, collections=['local_variables']) for x in out]
  with tf.control_dependencies([v.initializer for v in vs]):
    with tf.control_dependencies([queue.enqueue([v.read_value() for v in vs])]):
      return [v.read_value() for v in vs]


class TpuTest(parameterized.TestCase, test_utils.GapingTestCase):

  def setUp(self):
    super().setUp()

  def test_001_tpu_session(self):
    with self.session().as_default() as session:
      devices = session.list_devices()
      for device in devices:
        self.log(device)
      self.assertTrue(any([':TPU:' in device.name for device in devices]))
    self.assertTrue(self.is_tpu_available())

  def test_002_tpu_graph(self):
    with tf.Graph().as_default(), self.session().as_default() as session:
      self.assertNotEqual(session.graph, self.cached_session().graph)

  def test_003_add_tpu_cpu(self):
    with tf.Graph().as_default():
      self.assertEqual(3, self.evaluate(tf.add(1, 2)))

  def tpu_core_count(self):
    return 8

  def tpu_device_assignment(self):
    return wrapper.get_core_assignment(list(range(self.tpu_core_count())), topology=self.topology)

  def shard(self, *args, **kws):
    device_assignment = self.tpu_device_assignment()
    return wrapper.tpu_shard(*args, device_assignment=device_assignment, **kws)

  def test_003_add_tpu_cores(self):
    with tf.Graph().as_default():
      self.assertAllEqual([[3., 3., 3., 3., 3., 3., 3., 3.]],
          self.evaluate(self.shard(lambda: tf.add(1, 2))))

  def test_004_permute(self):
    with tf.Graph().as_default():
      inputs = [[tf.constant([x+1], dtype=tf.float32) for x in range(8)]]
      def on_cores(tensor):
        return tpu.tpu_ops.collective_permute(tensor,
            [[0, 1], [1, 0], [2, 3], [3, 2], [4, 5], [5, 4], [6, 7], [7, 6]])
      self.assertAllEqual([[[2.], [1.], [4.], [3.], [6.], [5.], [8.], [7.]]],
          self.evaluate(self.shard(on_cores, inputs=inputs)))

  def test_005_queue(self):
    with tf.Graph().as_default():
      image_path = tf1.placeholder(shape=(None), dtype=tf.string)
      image_label = tf1.placeholder(shape=(None), dtype=tf.int64)
      queue = tf.queue.RandomShuffleQueue(capacity=1_000_000, dtypes=[tf.string, tf.int64], shapes=[(), ()], min_after_dequeue=0, shared_name='inqueue')
      size_op = queue.size()
      enqueue_op = queue.enqueue_many([image_path, image_label])
      dequeue_op = queue.dequeue()
      requeue_op = requeue(queue)
      with self.session().as_default() as sess:
        if self.evaluate(size_op) <= 0:
          self.log('Enqueueing...')
          self.evaluate(enqueue_op, feed_dict={image_path: ['item2', 'item1', 'item4', 'item3'], image_label: [0, 1, 1, 0] })
        for i in range(10):
          self.log(self.evaluate(requeue_op))
          self.log('There are now %s items in the queue', self.evaluate(size_op))


#from gaping.models.inception_test import *

if __name__ == "__main__":
  import gaping.wrapper
  with gaping.wrapper.patch_tensorflow():
    tf.test.main()

