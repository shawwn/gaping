import os
import sys
sys.path += [os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')]

import tensorflow as tf
import tensorflow.compat.v1 as tf1

from absl import flags
from absl.testing import parameterized

from gaping import test_utils

from tensorflow.python.ops import critical_section_ops
from tensorflow.python.ops import resource_variable_ops


class CriticalSectionTest(parameterized.TestCase, test_utils.GapingTestCase):

  def setUp(self):
    super().setUp()

  def test_001_create_critical_section(self):
    cs = critical_section_ops.CriticalSection(shared_name="cs")
    v = resource_variable_ops.ResourceVariable(0.0, name="v")

    def fn(a, b):
      c = v.value()
      with tf.control_dependencies([c]):
        nv = v.assign_add(a * b)
        with tf.control_dependencies([nv]):
          return tf.identity(c)

    num_concurrent = 100
    r = [cs.execute(lambda: fn(1.0, 2.0)) for _ in range(num_concurrent)]
    self.evaluate(v.initializer)
    r_value = self.evaluate(r)
    print(list(sorted(r_value)))
    self.assertAllClose([2.0 * i for i in range(num_concurrent)],
                        sorted(r_value))
  


if __name__ == "__main__":
  import gaping.wrapper
  with gaping.wrapper.patch_tensorflow():
    tf.test.main()


