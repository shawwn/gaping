import os
import gin
from tensorflow.python.platform import test
mock = test.mock
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1

from gaping import wrapper


class GapingTestCase(tf.test.TestCase):
  """Base class for test cases."""

  def __init__(self, *args, **kws):
    super().__init__(*args, **kws)
    self._cached_session = None

  def _ClearCachedSession(self):
    if self._cached_session is not None:
      self._cached_session.close()
      self._cached_session = None

  def session(self, graph=None, config=None):
    if graph is None:
      graph = tf1.get_default_session()
    session = wrapper.clone_session(self.cached_session(), graph=graph, config=config)
    return session

  def cached_session(self):
    if self._cached_session is None:
      self._cached_session = wrapper.create_session()
    return self._cached_session

  def log(self, message, *args, **kws):
    tf1.logging.info(message, *args, **kws)

  def bucket_path(self, *parts):
    base = os.environ.get('MODEL_BUCKET') or os.environ['TPU_BUCKET']
    return os.path.join(base, *parts)

  def setUp(self):
    super(GapingTestCase, self).setUp()
    # Create the cached session.
    self.cached_session()
    # Clear the gin cofiguration.
    gin.clear_config()

