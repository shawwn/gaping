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

from tensorflow.python.data.experimental.ops import prefetching_ops
from tensorflow.python.data.ops import dataset_ops


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
    return self.skipTest("Takes a very long time")
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


  def test_004_recursiveCriticalSectionAccessIsIllegalSameSharedName(self):
    # This does not work properly in eager mode.  Eager users will
    # just hit a deadlock if they do this.  But at least it'll be easier
    # to debug.
    cs = critical_section_ops.CriticalSection(shared_name="cs")
    cs_same = critical_section_ops.CriticalSection(shared_name="cs")
    add = lambda x: x + 1
    def fn(x):
      return cs_same.execute(lambda: add(x))

    with self.assertRaisesRegex(
        ValueError, r"attempts to directly access the CriticalSection in which"):
      cs.execute(lambda: fn(1.0))


  def test_005_insideFunction(self):
    cs = critical_section_ops.CriticalSection()
    with tf.device("/device:TPU:0" if self.is_tpu_available() else "/device:CPU:0"):
      v = resource_variable_ops.ResourceVariable(1, name="test_000_insideFunction_v")
    def fn():
      return v.read_value()

    # map() creates a TensorFlow function.
    ds = dataset_ops.Dataset.range(1)
    if self.is_tpu_available():
      ds = (ds.apply(prefetching_ops.copy_to_device("/device:TPU:0"))
            .map(lambda _: cs.execute(fn)))
    else:
      ds = ds.map(lambda _: cs.execute(fn))

    def get_first():
      if context.executing_eagerly():
        return self.evaluate(dataset_ops.make_one_shot_iterator(ds).get_next())
      itr = dataset_ops.make_initializable_iterator(ds)
      self.evaluate([v.initializer, itr.initializer])
      return self.evaluate(itr.get_next())

    self.assertEqual(1, get_first())

if __name__ == "__main__":
  import gaping.wrapper
  with gaping.wrapper.patch_tensorflow():
    tf.test.main()


