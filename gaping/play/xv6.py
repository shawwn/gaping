import tensorflow as tf

from tensorflow.python.data.ops import optional_ops
from tensorflow.python.ops import list_ops

from argparse import Namespace as NS

# Registered kernels:
#   device='CPU'; key_dtype in [DT_INT32]; value_dtype in [DT_DOUBLE]
#   device='CPU'; key_dtype in [DT_INT32]; value_dtype in [DT_FLOAT]
#   device='CPU'; key_dtype in [DT_INT32]; value_dtype in [DT_INT32]
#   device='CPU'; key_dtype in [DT_INT64]; value_dtype in [DT_BOOL]
#   device='CPU'; key_dtype in [DT_INT64]; value_dtype in [DT_DOUBLE]
#   device='CPU'; key_dtype in [DT_INT64]; value_dtype in [DT_FLOAT]
#   device='CPU'; key_dtype in [DT_INT64]; value_dtype in [DT_INT32]
#   device='CPU'; key_dtype in [DT_INT64]; value_dtype in [DT_INT64]
#   device='CPU'; key_dtype in [DT_INT64]; value_dtype in [DT_VARIANT]
#   device='CPU'; key_dtype in [DT_STRING]; value_dtype in [DT_BOOL]
#   device='CPU'; key_dtype in [DT_STRING]; value_dtype in [DT_DOUBLE]
#   device='CPU'; key_dtype in [DT_STRING]; value_dtype in [DT_FLOAT]
#   device='CPU'; key_dtype in [DT_STRING]; value_dtype in [DT_INT32]
#   device='CPU'; key_dtype in [DT_STRING]; value_dtype in [DT_INT64]

class Table(tf.lookup.experimental.DenseHashTable):
  def __init__(self, key_dtype, value_dtype, default_value, empty_key, deleted_key,
      initial_num_buckets = 4,
      name="MutableDenseHashTable",
      checkpoint=False,
      shared_name=None,
      ):
    self._shared_name2 = shared_name
    super().__init__(key_dtype, value_dtype, default_value, empty_key, deleted_key,
        initial_num_buckets, name, checkpoint)
  
  def _create_resource(self):
    if self._shared_name is None:
      self._shared_name = self._shared_name2
    print('TKTK', self._shared_name)
    return super()._create_resource()


import functools

Int64Table = functools.partial(Table,
    key_dtype=tf.int64,
    value_dtype=tf.int64,
    default_value=0,
    empty_key=0,
    deleted_key=-1)

VariantTable = functools.partial(Table,
    key_dtype=tf.int64,
    value_dtype=tf.variant,
    empty_key=0,
    deleted_key=-1)

def tu64(x):
  x = tf.convert_to_tensor(x)
  if x.dtype == tf.uint64:
    return x
  else:
    return tf.cast(x, tf.uint64)

def ti64(x):
  x = tf.convert_to_tensor(x)
  if x.dtype == tf.int64:
    return x
  else:
    return tf.cast(x, tf.int64)

def tf_io_encode_raw(tokens, dtype=None):
  if dtype is not None:
    tokens = tf.cast(tokens, dtype)
  unit_size = tokens.dtype.size
  total_size = tf.size(tokens, out_type=tf.int64) * unit_size
  serialized = tf.serialize_tensor(tokens)
  serialized_size = tf.size(tf.strings.bytes_split(serialized), out_type=tf.int64)
  offset = serialized_size - total_size
  return tf.strings.substr(serialized, offset, -1)


def optional(v):
  if isinstance(v, optional_ops._OptionalImpl):
    return v
  return optional_ops.Optional.from_value(v)

def optional_nil(dtype, shape=()):
  spec = tf.TensorSpec(shape=shape, dtype=dtype)
  return optional_ops.Optional.none_from_structure(spec)

def optional_ref(v):
  return optional(v)._variant_tensor

def optional_opt(v, dtype, shape=()):
  spec = tf.TensorSpec(shape=shape, dtype=dtype)
  return optional_ops._OptionalImpl(v, spec)


KERNBASE =  0x80000000
PHYSTOP = (KERNBASE + 128*1024*1024)
PGSIZE = 4096
PGSHIFT = 12

#define PGROUNDUP(sz)  (((sz)+PGSIZE-1) & ~(PGSIZE-1))
#define PGROUNDDOWN(a) (((a)) & ~(PGSIZE-1))

def PGROUNDUP(sz):
  #sz = ti64(sz)
  sz = tf.convert_to_tensor(sz)
  lh = ti64(sz) + ti64(PGSIZE - 1)
  rh = tf.bitwise.invert(tu64(PGSIZE-1))
  out = tf.bitwise.bitwise_and(tu64(lh), rh)
  return tf.cast(out, sz.dtype)


def PGROUNDDOWN(sz):
  sz = tf.convert_to_tensor(sz)
  #sz = ti64(sz)
  lh = tu64(sz)
  rh = tf.bitwise.invert(tu64(PGSIZE-1))
  out = tf.bitwise.bitwise_and(tu64(lh), rh)
  return tf.cast(out, sz.dtype)

from contextlib import contextmanager, ExitStack

@contextmanager
def dep(*ops):
  with ExitStack() as stack:
    for op in ops:
      if not isinstance(op, (tuple, list)):
        op = [op]
      stack.enter_context(tf.control_dependencies(op))
    yield

def fori(pa_start, pa_stop, pa_step, fn, *args, parallel_iterations=1, **kws):
  # For a positive step, the contents of a range r are determined by the formula r[i] = start + step*i where i >= 0 and r[i] < stop.
  #
  # For a negative step, the contents of the range are still determined by the formula r[i] = start + step*i, but the constraints are i >= 0 and r[i] > stop.
  pa_start = tf.convert_to_tensor(pa_start)
  pa_stop = tf.convert_to_tensor(pa_stop, dtype=pa_start.dtype)
  pa_step = tf.convert_to_tensor(pa_step, dtype=pa_start.dtype)
  def r(i):
    return pa_start + pa_step*i
  def while_cond(i, *xs):
    return tf.where(pa_step > 0,
        tf.logical_and(i >= 0, r(i) < pa_stop),
        tf.logical_and(i >= 0, r(i) > pa_stop))
  def while_body(i, *xs):
    ys = fn(r(i), *xs)
    if not isinstance(ys, (tuple, list)):
      ys = [ys]
    if len(ys) > len(xs):
      with dep(ys[len(xs):]):
        return [tf.identity(x) for x in [i+1, *ys[0:len(xs)]]]
    else:
      return [i+1, *ys]
  with dep(tf.assert_positive( tf.abs(pa_step) )):
    return tf.while_loop(
        while_cond,
        while_body,
        [tf.convert_to_tensor(0, dtype=pa_start.dtype)] + list(args),
        parallel_iterations=parallel_iterations,
        **kws)


if 'G' not in globals() or getattr(G, 'graph', None) != tf.get_default_graph():
  G = NS()
  G.graph = tf.get_default_graph()

if not hasattr(G, 'ptr'):
  G.ptr = Int64Table(
      shared_name="xv6/ptr")

if not hasattr(G, 'mem'):
  G.mem = VariantTable(
      shared_name="xv6/mem",
      default_value=optional_ref(optional_nil(tf.string)),
      )

def forpage(start, end, fn, *args, unset=None, **kws):
  start = PGROUNDDOWN(start)
  end = PGROUNDUP(end)
  def body(addr, *args):
    addr_end = addr + PGSIZE
    o = optional_opt(G.mem.lookup(addr), tf.string)
    def cond_set():
      page = o.get_value()
      return fn(addr, addr_end, page, *args)
    def cond_unset():
      nonlocal unset
      if callable(unset):
        unset = unset(addr, addr_end, *args)
      return fn(addr, addr_end, unset, *args)
    return tf.cond(o.has_value(),
        cond_set,
        cond_unset)
  return fori(start, end, PGSIZE, body, *args)

def restrict(*xs):
  addr = max(xs[0::2])
  end = min(xs[1::2])
  return addr, end

def pagebuf(page):
  if page.dtype == tf.string:
    return tf.io.decode_raw(page, tf.uint8)
  assert page.dtype == tf.uint8
  return page

from .. import tf_api as api

def pagearr(page):
  buf = pagebuf(page)
  ta = api.tf_array_new(tf.uint8)
  ta = api.tf_array_extend(ta, buf)
  return ta

def memset(p, byte, size):
  byte = tf.constant(byte, tf.uint8)
  e = p + size
  def body(addr, end, page):
    if page is None:
      page = tf.constant(b'\xfe' * PGSIZE)
    arr = pagearr(page)
    def inner(x, i):
      return tf.where(
          tf.logical_and(
            tf.greater_equal(addr + i, p),
            tf.less(addr + i, e)),
          byte,
          x)
    arr = api.tf_array_map(arr, inner, out_dtype=tf.uint8)
    page = tf_io_encode_raw(arr.stack())
    return G.mem.insert(addr, optional_ref(page))
  return forpage(p, p+size, body)

def memcpy(dst, src, size):
  dst_end = dst + size
  src = tf.convert_to_tensor(src)
  src = pagebuf(src)
  def body(addr, end, page):
    if page is None:
      page = tf.constant(b'\xfe' * PGSIZE)
    arr = pagearr(page)
    def inner(x, i):
      return tf.cond(
          tf.logical_and(
            tf.greater_equal(addr + i, dst),
            tf.less(addr + i, dst_end)),
          lambda: src[(addr - dst) + i],
          lambda: x)
    arr = api.tf_array_map(arr, inner, out_dtype=tf.uint8)
    page = tf_io_encode_raw(arr.stack())
    return G.mem.insert(addr, optional_ref(page))
  return forpage(dst, dst_end, body)

def memread(dst, size, dtype=tf.string):
  dst = ti64(dst)
  size = ti64(size)
  dst_end = dst + size
  ta = api.tf_array_new(tf.uint8)
  def body(addr, end, page, ta):
    if page is None:
      page = tf.constant(b'\xfe' * PGSIZE)
    buf = pagebuf(page)
    ta = api.tf_array_extend(ta, buf)
    return ta
  _, to = forpage(dst, dst_end, body, ta)
  out = to.stack()
  off = dst - PGROUNDDOWN(dst)
  out = out[off:off+size]
  if dtype == tf.string:
    out = tf_io_encode_raw(out)
  else:
    out = tf.bitcast(out, dtype)
  return out
  
end = KERNBASE

def kinit():
  # initlock(&kmem.lock, "kmem");
  # freerange(end, (void*)PHYSTOP);
  return freerange(end, PHYSTOP)

def freerange(pa_start, pa_end):
  return fori(pa_start, pa_end, PGSIZE, kfree)

def kfree(p):
  with tf.control_dependencies([tf.assert_equal(p, PGROUNDDOWN(p))]):
    return tf.no_op()

