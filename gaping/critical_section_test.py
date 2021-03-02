import os
import sys
sys.path += [os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')]

import itertools
import tqdm

import tensorflow as tf
import tensorflow.compat.v1 as tf1

from absl import flags
from absl.testing import parameterized

from gaping import test_utils

from tensorflow.python.ops import critical_section_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function


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
  

  @parameterized.named_parameters(
      ("Inner%sOuter%s" % (inner, outer), inner, outer)
      for (inner, outer) in itertools.product(*([(False, True)] * 2)))
  def test_002_critical_section_with_control_flow(self, outer_cond, inner_cond):
    cs = critical_section_ops.CriticalSection(shared_name="cs")
    v = resource_variable_ops.ResourceVariable(0.0, name="v")
    num_concurrent = 100


    # pylint: disable=cell-var-from-loop
    def fn(a, b):
      c = v.read_value()
      def true_fn():
        with tf.control_dependencies([c]):
          nv = v.assign_add(a * b)
          with tf.control_dependencies([nv]):
            return tf.identity(c)
      return tf.cond(
          tf.identity(inner_cond), true_fn, lambda: c)

    def execute():
      return cs.execute(lambda: fn(1.0, 2.0))

    r = [
        tf.cond(tf.identity(outer_cond),
                              execute,
                              v.read_value)
        for _ in range(num_concurrent)
    ]
    # pylint: enable=cell-var-from-loop

    self.evaluate(v.initializer)
    r_value = self.evaluate(r)
    if inner_cond and outer_cond:
      self.assertAllClose([2.0 * i for i in range(num_concurrent)],
                          sorted(r_value))
    else:
      self.assertAllClose([0] * num_concurrent, r_value)
    
  def test_003_criticalSectionInParallelDoesntDeadlockOnError(self):
    # No eager mode execution of this test because eager does not
    # run fn() in parallel, which is where the deadlock could
    # potentially occur (in graph mode).
    cs = critical_section_ops.CriticalSection(shared_name="cs")
    v = resource_variable_ops.ResourceVariable(0.0, name="v")

    def fn(i):
      error = tf.Assert((i % 2) == 1, ["Error"])
      with tf.control_dependencies([error]):
        return v.read_value()

    num_concurrent = 2

    @def_function.function(autograph=False)
    def run_concurrently():
      return [cs.execute(lambda: fn(i)) for i in range(num_concurrent)]

    if not context.executing_eagerly():
      run_concurrently = run_concurrently()

    self.evaluate(v.initializer)
    for _ in tqdm.trange(10):
      with self.assertRaisesOpError("Error"):
        if context.executing_eagerly():
          run_concurrently()
        else:
          self.evaluate(run_concurrently)

if __name__ == "__main__":
  import gaping.wrapper
  with gaping.wrapper.patch_tensorflow():
    tf.test.main()


