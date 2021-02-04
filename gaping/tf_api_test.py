import tensorflow as tf
import os
import tqdm

from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.lib.io import python_io
from tensorflow.python.platform import test
from tensorflow.python.tpu import datasets
from tensorflow.python.training import server_lib
from tensorflow.python.util import compat

from gaping import test_utils
from gaping import wrapper
from gaping.tf_api import *


class TFApiTest(test_utils.GapingTestCase):

  def setUp(self):
    super().setUp()

  def test_001_basic(self):
    with self.session() as session:
      ta = tf_array_new(tf.int32)
      #self.assertAllClose([], self.evaluate( ta.stack() ) )
      ta = tf_array_push(ta, 42)
      self.assertAllClose([42], self.evaluate( ta.stack() ) )
      ta = tf_array_push(ta, 99)
      self.assertAllClose([42, 99], self.evaluate( ta.stack() ) )
      ta = tf_array_extend(ta, tf.range(3))
      self.assertAllClose([42, 99, 0, 1, 2], self.evaluate( ta.stack() ) )
      self.assertEqual(5, self.evaluate( tf_len(ta)) )
      ta = tf_array_map(ta, lambda x: x + 1)
      self.assertAllClose([43, 100, 1, 2, 3], self.evaluate( ta.stack() ) )
      ta = tf_array_mappend(ta, lambda x: [x, x])
      self.assertAllClose([43, 43, 100, 100, 1, 1, 2, 2, 3, 3],
          self.evaluate( ta.stack() ) )
      ta = tf_array_map(ta, lambda x, i: i)
      self.assertAllClose(list(range(10)), self.evaluate( ta.stack() ) )


if __name__ == "__main__":
  import gaping.wrapper
  with gaping.wrapper.patch_tensorflow():
    tf.test.main()


