import os
import gin
from tensorflow.python.platform import test
mock = test.mock
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1


class GapingTestCase(tf.test.TestCase):
  """Base class for test cases."""

  def setUp(self):
    super(GapingTestCase, self).setUp()
    # Clear the gin cofiguration.
    gin.clear_config()

  def log(self, message, *args, **kws):
    tf1.logging.info(message, *args, **kws)

  def bucket_path(self, *parts):
    base = os.environ.get('MODEL_BUCKET') or os.environ['TPU_BUCKET']
    return os.path.join(base, *parts)

