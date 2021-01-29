import tensorflow.compat.v1 as tf
from tensorflow.python.util import nest

# fall back to while loops for pfor
from tensorflow.python.ops.parallel_for.pfor import flags as pfor_flags
pfor_flags.FLAGS.op_conversion_fallback_to_while_loop = True

from tensorflow.python.ops.parallel_for import control_flow_ops as pfor_control_flow_ops

from tensorflow.python.framework import errors_impl

def loop_n(thunk, n, **kws):
  return pfor_control_flow_ops.pfor(thunk, iters=n, **kws)


class Queue:
  def __init__(self, dtypes, shapes=None, capacity=1_000_000, shuffle=False, min_after_dequeue=0, shared_name=None):
    self.shuffle = shuffle
    self.capacity = capacity
    # if shapes is None:
    #   shapes = [() for _ in range(len(dtypes))]
    self.dtypes = dtypes
    self.shapes = shapes
    self.min_after_dequeue = min_after_dequeue
    self.shared_name = shared_name
    if self.shuffle:
      self.queue = tf.queue.RandomShuffleQueue(capacity=self.capacity, dtypes=self.dtypes, shapes=self.shapes, min_after_dequeue=self.min_after_dequeue, shared_name=self.shared_name)
    else:
      self.queue = tf.queue.FIFOQueue(capacity=self.capacity, dtypes=self.dtypes, shapes=self.shapes, shared_name=self.shared_name)

  def enqueue_many(self, vals, name=None):
    return self.queue.enqueue_many(vals, name=name)

  def dequeue(self, n=None, transform=None, name="dequeue"):
    def inner(x):
      x = self.queue.dequeue(name=name)
      if transform is not None:
        x = transform(x)
      return x
    if n is None:
      return inner(0)
    else:
      return loop_n(inner, n)


def requeue(queue):
  out = queue.dequeue()
  vs = [tf.Variable(x, use_resource=True, trainable=False, collections=['local_variables']) for x in out]
  with tf.control_dependencies([v.initializer for v in vs]):
    with tf.control_dependencies([queue.enqueue([v.read_value() for v in vs])]):
      return [v.read_value() for v in vs]


class InputDataset:
  def __init__(self, dataset, transform=None):
    self.dataset = dataset
    self.started = False
    self.iterator = self.dataset.make_one_shot_iterator()
    self.feature_structure = self.iterator.get_next()
    self.flattened_inputs = nest.flatten(self.feature_structure)
    self.op = self.flattened_inputs
    if transform:
      self.op = transform(self.op)
  
  def __iter__(self):
    assert self.started is False, "Can't iterate multiple times"
    self.started = True
    sess = tf.get_default_session()
    while True:
      try:
        yield sess.run(self.op)
      except errors_impl.OutOfRangeError:
        break
