import gin
from tensorflow.python.platform import test
mock = test.mock
import numpy as np
import tensorflow as tf
import os


class GapingTestCase(tf.test.TestCase):
  """Base class for test cases."""

  def setUp(self):
    super(GapingTestCase, self).setUp()
    # Clear the gin cofiguration.
    gin.clear_config()

  def bucket_path(self, *parts):
    return os.path.join(os.environ['TPU_BUCKET'], *parts)

