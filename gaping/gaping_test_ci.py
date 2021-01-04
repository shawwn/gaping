import os
import sys
sys.path += [os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')]
import tensorflow as tf


from absl import flags
from absl.testing import parameterized

from gaping import test_utils
import tensorflow_hub as hub


class GapingTestCI(parameterized.TestCase, test_utils.GapingTestCase):

  def setUp(self):
    super().setUp()

  @parameterized.parameters([42,99])
  def testBasic(self, value):
    print('cwd: {!r}'.format(os.getcwd()))
    print('value: {}'.format(value))


if __name__ == "__main__":
  import gaping.wrapper
  with gaping.wrapper.patch_tensorflow():
    tf.test.main()

