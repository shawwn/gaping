import os
import sys
sys.path += [os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')]

import tensorflow as tf
import tensorflow.compat.v1 as tf1

from absl import flags
from absl.testing import parameterized

from gaping import test_utils
from gaping import wrapper

from gaping import tftorch as nn

import numpy as np
import tqdm

from tensorflow.python.training import gradient_descent

class OptimizerTest(parameterized.TestCase, test_utils.GapingTestCase):

  def setUp(self):
    super().setUp()

  def test_000_optimizer_tpu_session(self):
    return
    with self.session().as_default() as session:
      devices = session.list_devices()
      for device in devices:
        self.log(device)
      self.assertTrue(any([':TPU:' in device.name for device in devices]))

  def test_001_optimizer_basic(self):
    with self.session().as_default() as session:
      var0 = tf.Variable([1.0, 2.0], name='a_0')
      var1 = tf.Variable([3.0, 4.0], name='b_0')
      def loss():
        return 5 * var0 + 3 * var1
      global_step = tf.train.get_or_create_global_step()
      sgd_op = gradient_descent.GradientDescentOptimizer(3.0)
      self.evaluate(tf.global_variables_initializer())
      # Fetch params to validate initial values
      self.assertAllClose([1.0, 2.0], self.evaluate(var0))
      self.assertAllClose([3.0, 4.0], self.evaluate(var1))
      # Run 1 step of sgd through optimizer
      opt_op = sgd_op.minimize(loss, global_step, [var0, var1])
      self.evaluate(opt_op)
      # Validate updated params
      self.assertAllClose([-14., -13.], self.evaluate(var0))
      self.assertAllClose([-6., -5.], self.evaluate(var1))

if __name__ == "__main__":
  with wrapper.patch_tensorflow():
    tf.test.main()



