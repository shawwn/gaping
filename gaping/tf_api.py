import tensorflow.compat.v1 as tf

from . import tf_tools as tft
from . import tf_ext
from . import util

from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.framework import ops

import weakref
import inspect


def refv(name, **kws):
  shared_name = name.rsplit(':',1)[0]
  return tf_ext.RefVariable( name=name, shared_name=shared_name, handle_name=shared_name, unique_id=shared_name, **kws)

def tf_array_impl_p(ta):
  return isinstance(ta, (tensor_array_ops._GraphTensorArray, util.EasyDict))

def tf_array_impl(ta):
  if hasattr(ta, '_implementation'):
    ta = ta._implementation
  if not tf_array_impl_p(ta):
    raise ValueError("Not an array impl: {!r}".format(ta))
  return ta

def tf_array_dict(ta, **kws):
  if not isinstance(ta, dict):
    ta = tf_array_impl(ta)
    ta = vars(ta)
  ta = util.EasyDict(ta)
  for k, v in kws.items():
    ta[k] = v
  return ta

def tf_array_array(ta):
  if ta is None:
    return ta
  if isinstance(ta, dict):
    ta = ta.get('parent')
    if callable(ta):
      ta = ta()
  return ta

def tf_array_getattr(ta, attribute, *default):
  if ta is None:
    return default[0] if default else None
  ta = tf_array_dict(ta)
  if hasattr(ta, attribute):
    return getattr(ta, attribute)
  if hasattr(ta, '_' + attribute):
    return getattr(ta, '_' + attribute)
  ta = tf_array_array(ta)
  return getattr(ta, attribute, *default)

def tf_array_getter(fn, *use_default):
  fn = getattr(fn, '__name__', fn)
  if fn.startswith('tf_array_'):
    fn = fn.split('tf_array_', 1)[1]
  attr = fn
  def getter(ta, *default, **kws):
    result = tf_array_getattr(ta, attr, *(default or use_default), **kws)
    if isinstance(result, weakref.ref):
      result = result()
    return result
  getter.__qualname__ = getter.__name__ = 'tf_array_' + fn
  return getter

tf_array_attrs = 'dtype _infer_shape _dynamic_size handle flow _colocate_with _colocate_with_first_write_call _element_shape parent _prev?'.split()

from collections import OrderedDict

tf_array_defaults = OrderedDict(
    dtype=tf.int32,
    size=0,
    dynamic_size=True,
    clear_after_read=True,
    tensor_array_name=None,
    handle=None,
    flow=None,
    infer_shape=True,
    element_shape=None,
    colocate_with_first_write_call=True,
    name=None)

def tf_array_clone(ta, dtype=None):
  return tf_array_new(
    dtype=dtype if dtype is not None else tf_array_dtype(ta),
    size=0,
    dynamic_size=True,
    clear_after_read=True,
    tensor_array_name=None,
    handle=None,
    flow=None,
    infer_shape=True,
    element_shape=None,
    colocate_with_first_write_call=True,
    name=None)


for attr in tf_array_attrs:
  default = [None] if attr.endswith('?') else []
  attr = attr.lstrip('_').rstrip('?')
  globals()['tf_array_' + attr] = tf_array_getter(attr, *default)

def tf_array_set_prev(ta, prev):
  ta = tf_array_array(ta)
  prev = tf_array_array(prev)
  if prev is not None:
    prev = weakref.ref(prev)
  impl = tf_array_impl(ta)
  setattr(impl, '_prev', prev)
  return prev

def tf_array_info(ta, **replacements):
  tf_array_keys = []
  tf_array_values = []
  for attr in tf_array_attrs:
    default = [None] if attr.endswith('?') else []
    key = attr.rstrip('?')
    attr = key.lstrip('_')
    value = tf_array_getattr(ta, attr, *default)
    tf_array_keys.append(key)
    tf_array_values.append(value)
  info = util.EasyDict(zip(tf_array_keys, tf_array_values))
  for k, v in replacements.items():
    if '_' + k in tf_array_attrs or '_' + k + '?' in tf_array_attrs:
      k = '_' + k
    if v is None:
      if hasattr(info, k):
        del info[k]
    else:
      info[k] = v
  for k, v in list(info.items()):
    if v is None:
      del info[k]
  return info

def tf_array_new_args(*args, **kws):
  params = {}
  for v, k in zip(args, tf_array_defaults.keys()):
    params[k] = v
  for k, v in kws.items():
    assert k not in params, "You specified a key multiple times"
    params[k] = v
  for k, v in tf_array_defaults.items():
    if k not in params:
      params[k] = v
  return params

import functools

@functools.wraps(tf.TensorArray)
def tf_array_new(*args, **kws):
  params = tf_array_new_args(*args, **kws)
  array = tf.TensorArray(**params)
  return array

def tf_array_copy(ta, **replacements):
  info = tf_array_info(ta, **replacements)
  prev = tf_array_array(info)
  flow = tf_array_flow(info)
  new = tensor_array_ops.build_ta_with_new_flow(info, flow)
  tf_array_set_prev(new, ta)
  return new

from contextlib import contextmanager

@contextmanager
def tf_array_colocate(ta):
  with ops.colocate_with(tf_array_colocate_with(ta), ignore_existing=True):
    yield

def tf_array_write(ta, index, value, *, flow=None, handle=None):
  if flow is None:
    flow = tf_array_flow(ta)
  if handle is None:
    handle = tf_array_handle(ta)
  with tf_array_colocate(ta):
    dtype = tf_array_dtype(ta)
    value = tf.cast(value, dtype)
    next_flow = tf.raw_ops.TensorArrayWriteV3(
        handle=handle,
        index=index,
        value=value,
        flow_in=flow )
  return tf_array_copy(ta, flow=next_flow)

def tf_verify_indices_in_int32(x):
  indices = tf.cast( x, tf.int64 )
  minimum = tf.cast( 0, tf.int64)
  maximum = tf.cast( (1 << 31) - 1, tf.int64)
  ops = [
    tf.debugging.assert_greater_equal( indices, minimum ),
    tf.debugging.assert_less_equal( indices, maximum ),
    ]
  with tf.control_dependencies(ops):
    return tf.cast(x, tf.int32)

def tf_array_scatter(ta, indices, value, *, flow=None, handle=None):
  if flow is None:
    flow = tf_array_flow(ta)
  if handle is None:
    handle = tf_array_handle(ta)
  with tf_array_colocate(ta):
    dtype = tf_array_dtype(ta)
    indices = tf_verify_indices_in_int32(indices)
    value = tf.cast(value, dtype)
    next_flow = tf.raw_ops.TensorArrayScatterV3(
        handle=handle,
        indices=indices,
        value=value,
        flow_in=flow )
  return tf_array_copy(ta, flow=next_flow)

def tf_array_gather(ta, indices, *, flow=None, handle=None):
  if flow is None:
    flow = tf_array_flow(ta)
  if handle is None:
    handle = tf_array_handle(ta)
  with tf_array_colocate(ta):
    dtype = tf_array_dtype(ta)
    indices = tf_verify_indices_in_int32(indices)
    return tf.raw_ops.TensorArrayGatherV3(
        handle=handle,
        indices=indices,
        dtype=dtype,
        flow_in=flow )

def tf_array_size(ta, *, flow=None, handle=None):
  if flow is None:
    flow = tf_array_flow(ta)
  if handle is None:
    handle = tf_array_handle(ta)
  size_op = tf.raw_ops.TensorArraySizeV3(
      handle=handle,
      flow_in=flow )
  return size_op

def tf_array_push(ta, value, *, flow=None, handle=None):
  index = tf_array_size(ta, flow=flow, handle=handle)
  return tf_array_write(ta, index=index, value=value, flow=flow, handle=handle)

def tf_len(x):
  if isinstance(x, tf.TensorArray):
    x = tf_array_size(x)
    x = tf_i64(x)
    return x
  x = tf.convert_to_tensor(x)
  if x.dtype == tf.string:
    x = tf.strings.length(x)
    x = tf_i64(x)
    return x
  else:
    return tf.size(x, out_type=tf.int64)

def tf_len32(x):
  x = tf_len(x)
  x = tf_i32(x)
  return x

@functools.wraps(tf.convert_to_tensor)
def tf_tensor(x, dtype=None, name=None, preferred_dtype=None, dtype_hint=None):
  dtype = tf_dtype(dtype)
  preferred_dtype = tf_dtype(preferred_dtype)
  dtype_hint = tf_dtype(dtype_hint)
  return tf.convert_to_tensor(x, dtype=dtype, name=name, preferred_dtype=preferred_dtype, dtype_hint=dtype_hint)

tf_t = tf_tensor

def tf_dtype(t):
  if t is None:
    return t
  if hasattr(t, 'dtype'):
    t = t.dtype
  if isinstance(t, str):
    t = t.replace('boolean', 'bool')
    t = t.replace('string', 's')
    t = t.replace('str', 's')
    t = t.replace('uint', 'u')
    t = t.replace('int', 'i')
    t = t.replace('float', 'f')
    t = t.replace('i', 'int')
    t = t.replace('u', 'uint')
    t = t.replace('f', 'float')
    t = t.replace('s', 'string')
    t = tf.dtypes.as_dtype(t)
  if isinstance(t, tf.dtypes.DType):
    return t
  #return tf.convert_to_tensor(t).dtype
  raise ValueError("Unknown dtype: {!r}".format(t))

def tf_dtypes(dtypes):
  if isinstance(dtypes, str):
    specs = dtypes.split()
    dtypes = []
    for s in specs:
      if s in ['f', 'float']:
        s = 'bf16 f16 f32 f64'
      elif s in ['u', 'unsigned']:
        s = 'u8 u16 u32 u64'
      elif s in ['signed']:
        s = 'i8 i16 i32 i64'
      elif s in ['i', 'int', 'integer']:
        s = 'u8 u16 u32 u64 i8 i16 i32 i64'
      dtypes.extend(s.split())
  if not isinstance(dtypes, (tuple, list)):
    dtypes = [dtypes]
  return [tf_dtype(t) for t in dtypes]

def tf_is_type(dtypes, *args):
  dtypes = [tf_dtype(t) for t in dtypes]
  for x in args:
    x = tf.convert_to_tensor(x)
    t = tf_dtype(x)
    if t not in dtypes:
      return False
  return True

def tf_to(dtype, *args, **kws):
  dtype = tf_dtype(dtype)
  if len(args) == 1:
    return tf.cast(*args, dtype=dtype, **kws)
  else:
    return tuple([tf.cast(x, dtype=dtype, **kws) for x in args])

import functools

def tf_caster(dtype):
  def tf_casting_to(*args, **kws):
    return tf_to(dtype, *args, **kws)
  return tf_casting_to

# create casters for all permutations of tf_{i,u,f}{8,16,32,64}
for kind in 'i u f'.split():
  for size in '8 16 32 64'.split():
    globals()['tf_'+kind+size] = tf_caster(kind+size)
    globals()['tf_'+kind+size].__qualname__ = 'tf_'+kind+size

def tf_irange(start, limit=None, delta=1, dtype=tf.int64, name='range'):
  if limit is None:
    start, limit = 0, start
  start, limit, delta = tf_i64(start, limit, delta)
  out = tf.range(start, limit, delta, dtype=tf.int64, name=name)
  return tf_to(dtype, out)

def tf_array_indices(ta):
  return tf_irange(tf_len32(ta))

def tf_array_extend(ta, *args):
  for value in args:
    start = tf_len(ta)
    count = tf_len(value)
    indices = tf_irange(start, start + count)
    ta = tf_array_scatter(ta, indices=indices, value=value)
  return ta

def tf_string_split(input, delimiter='\n', return_all=False):
  indices, results, shape = tf.raw_ops.StringSplit(input=input, delimiter=delimiter)
  if return_all:
    return indices, results, shape
  else:
    return results

def tf_string_splits(input, delimiters=[' ', '\n']):
  for delim in delimiters:
    input = tf_string_split(input, delim)
  return input

def tf_array_fn_needs_index(f):
  sig = inspect.signature(f)
  return len(sig.parameters) > 1

def tf_array_map(ta, f, out_dtype=None, *, start=0, parallel_iterations=96, **kws):
  def _while_condition(*args, **kws):
    return tf.convert_to_tensor(True)
  def _while_body(i, a_i, a_o):
    k = tf_i32( i )
    x = a_i.read(k)
    if tf_array_fn_needs_index(f):
      y = f(x, i)
    else:
      y = f(x)
    return i + 1, a_i, tf_array_write(a_o, k, y)
    #return i + 1, tf_array_write(a, k, x) #  Could not write to TensorArray index 0 because it has already been read.
  ta_out = tf.TensorArray(
      dtype=tf_dtype(out_dtype) or tf_array_dtype(ta),
      size=tf_len32(ta))
  _, ta, ta_out = tf.while_loop(
      _while_condition, _while_body, (
          tf_i64(start), 
          ta,
          ta_out,
      ),
      parallel_iterations=parallel_iterations,
      maximum_iterations=tf_len32(ta),
      **kws
      )
  #return _, ta, ta_out
  return ta_out


def tf_array_mappend(ta, f, out_dtype=None, *, start=0, **kws):
  parallel_iterations=1
  def _while_condition(*args, **kws):
    return tf.convert_to_tensor(True)
  def _while_body(i, a_i, a_o):
    k = tf_i32( i )
    x = a_i.read(k)
    if tf_array_fn_needs_index(f):
      y = f(x, i)
    else:
      y = f(x)
    return i + 1, a_i, tf_array_extend(a_o, y)
    #return i + 1, tf_array_write(a, k, x) #  Could not write to TensorArray index 0 because it has already been read.
  ta_out = tf_array_new(tf_dtype(out_dtype) or tf_array_dtype(ta))
  _, ta, ta_out = tf.while_loop(
      _while_condition, _while_body, (
          tf_i64(start), 
          ta,
          ta_out,
      ),
      parallel_iterations=parallel_iterations,
      maximum_iterations=tf_len32(ta),
      **kws
      )
  #return _, ta, ta_out
  return ta_out


def tf_set_shape(x, shape):
  x.set_shape(shape)
  return x

# run( tensor_array_ops.build_ta_with_new_flow( util.EasyDict(dtype=tf.string, handle=wrote.handle, infer_shape=True, _colocate_with_first_write_call=True, _dynamic_size=True, _colocate_with=[], _element_shape=shape, _infer_shape=True), wrote.write(1,'baz').flow ).size() )
