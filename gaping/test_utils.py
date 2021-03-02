import os
import gin
from tensorflow.python.platform import test
mock = test.mock
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1

from gaping import wrapper
import gaping.driver

from tensorflow.python.eager import context
from tensorflow.python.framework import ops

_cached_tpu_topology = None

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
      graph = ops.get_default_graph()
    session = wrapper.clone_session(self.cached_session(), graph=graph, config=config)
    return session

  @property
  def topology(self):
    return self.cached_session().driver.topology

  def cached_session(self, interactive=False):
    if self._cached_session is None:
      driver = gaping.driver.new(interactive=interactive)
      self._cached_session = driver.session
    return self._cached_session

  def evaluate(self, tensors, **kws):
    """Evaluates tensors and returns numpy values.

    Args:
      tensors: A Tensor or a nested list/tuple of Tensors.

    Returns:
      tensors numpy values.
    """
    if context.executing_eagerly():
      raise NotImplementedError()
      #return self._eval_helper(tensors)
    else:
      sess = ops.get_default_session()
      if sess is None:
        with self.session() as sess:
          return sess.run(tensors, **kws)
      else:
        return sess.run(tensors, **kws)

  def log(self, message, *args, **kws):
    tf1.logging.info(message, *args, **kws)

  def bucket_path(self, *parts):
    base = os.environ.get('MODEL_BUCKET') or os.environ['TPU_BUCKET']
    return os.path.join(base, *parts)

  def setUp(self):
    super().setUp()
    # Create the cached session.
    self.cached_session()
    # Clear the gin cofiguration.
    gin.clear_config()

