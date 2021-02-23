from .. import driver as driver_lib
from .. import tf_tools as tft

import tensorflow as tf

from tensorflow.python.framework import ops

from . import sequence

import numpy as np

from contextlib import contextmanager, ExitStack

def get_minimum_sequence(sequences, minimum=None):
  if minimum is None:
    minimum = tf.constant(np.iinfo(np.int64).max, dtype=tf.int64)
  minimum = ti64(minimum)
  for seq in sequences:
    value = seq.get()
    minimum = tf.minimum(minimum, value)
  return minimum

def get_sequences_for(*processors):
  sequences = []
  for processor in processors:
    sequences.append(processor.get_sequence())
  return sequences

def dep(op):
  if not isinstance(op, (tuple, list)):
    op = [op]
  return tf.control_dependencies(op)
  # with ExitStack() as stack:
  #   for op in ops:
  #     if not isinstance(op, (tuple, list)):
  #       op = [op]
  #     stack.enter_context(tf.control_dependencies(op))
  #   yield

def panic(condition, msg, *args):
  return tf.debugging.Assert(condition, [msg, *args])

def check(x, condition, msg, *args):
  if callable(condition):
    condition = condition(x)
  with dep(panic(condition, msg, *args)):
    return tf.identity(x)

def tf64(x):
  x = tf.convert_to_tensor(x)
  if x.dtype == tf.float64:
    return x
  else:
    return tf.cast(x, tf.float64)


def ti64(x):
  x = tf.convert_to_tensor(x)
  if x.dtype == tf.int64:
    return x
  else:
    return tf.cast(x, tf.int64)


def ti32(x):
  x = tf.convert_to_tensor(x)
  if x.dtype == tf.int32:
    return x
  else:
    return tf.cast(x, tf.int32)


from tensorflow.python.data.experimental.ops import sleep as sleep_lib

def sleep(seconds):
  def on_cpu(seconds):
    with ops.init_scope():
      usec = ti64(seconds * 1e6)
      dset = tf.data.Dataset.from_tensor_slices([0])
      dset = dset.repeat()
      dset = dset.apply( sleep_lib.sleep( usec ) )
      it = dset.make_one_shot_iterator()
    # start = tft.tpu_nanos()
    # with dep(it.get_next()):
    #   elapsed = tft.tpu_nanos() - start
    #   return elapsed
    if False:
      return it.get_next()
    else:
      with dep(it.get_next()):
        return tft.tpu_nanos()
  return tft.tpu_cpu(on_cpu, seconds)


# fall back to while loops for pfor
from tensorflow.python.ops.parallel_for.pfor import flags as pfor_flags
pfor_flags.FLAGS.op_conversion_fallback_to_while_loop = True

from tensorflow.python.ops.parallel_for import control_flow_ops as pfor_control_flow_ops


# def loop_n(thunk, dtypes, n, **kws):
#   parallel_iterations = kws.pop('parallel_iterations', 1)
#   return pfor_control_flow_ops.for_loop(thunk, dtypes, iters=n, parallel_iterations=parallel_iterations, **kws)


def loop_n(thunk, n, initial_value=0, **kws):
  parallel_iterations = kws.pop('parallel_iterations', 1)
  if callable(initial_value):
    initial_value = initial_value()
  initial_value = tf.convert_to_tensor(initial_value)
  def body(i, value):
    y = thunk(value)
    with dep(y):
      return i + 1, tf.cast(y, dtype=initial_value.dtype)
  i, value = tf.while_loop(
      lambda *args: True,
      body,
      loop_vars=(tf.cast(0, tf.int32), initial_value),
      parallel_iterations=parallel_iterations,
      maximum_iterations=n,
      **kws,
      )
  return value


def loop_til(cond, body, *args, **kws):
  parallel_iterations = kws.pop('parallel_iterations', 1)
  # def body(i, *args):
  #   return i + 1, *args
  out = tf.while_loop(
      cond,
      body,
      loop_vars=tuple([tf.convert_to_tensor(x) for x in args]),
      parallel_iterations=parallel_iterations,
      **kws,
      )
  return out

def check_every(seconds, still_running, final_result):
  start = tft.tpu_nanos()
  def cond(i, result, running):
    return running
  def body(i, result, running):
    elapsed = tf64(tft.tpu_nanos() - start) / 1e9
    running = still_running(elapsed)
    result = final_result(elapsed)
    #running, result = pred(elapsed)
    def yes():
      with dep(sleep(seconds)):
        return i + 1, result, running
    def no():
      return i + 1, result, running
    return tf.cond(running,
        yes,
        no)
  i, result, running = loop_til(cond, body, ti32(0), False, True)
  return result

