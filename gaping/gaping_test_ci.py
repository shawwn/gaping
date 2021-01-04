import os
import sys
sys.path += [os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')]

import tensorflow as tf
import tensorflow.compat.v1 as tf1

from absl import flags
from absl.testing import parameterized

from gaping import test_utils
import tensorflow_hub as hub


class GapingTestCI(parameterized.TestCase, test_utils.GapingTestCase):

  def setUp(self):
    super().setUp()

  def test_01_info(self):
    self.log('cwd: %s', os.getcwd())
    self.log('TF version: %s', tf.__version__)

  @parameterized.parameters([42,99])
  def test_02_basic_parameterized(self, value):
    self.log('parameterized value: %s', value)


if __name__ == "__main__":
  import gaping.wrapper
  with gaping.wrapper.patch_tensorflow():
    tf.test.main()

