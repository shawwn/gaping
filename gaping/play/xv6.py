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
  if getattr(tokens, 'dtype', None) == tf.string:
    return tokens
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

def loop(initial, getter, cond, fn, *args, incr=None, parallel_iterations=96, **kws):
  if getter is None:
    getter = lambda i: i
  if incr is None:
    incr = lambda i: i+1
  elif not callable(incr):
    inc = incr
    incr = lambda i: i+inc
  def while_cond(i, *xs):
    return cond(getter(i), *xs)
  def while_body(i, *xs):
    ys = fn(getter(i), *xs)
    if not isinstance(ys, (tuple, list)):
      ys = [ys]
    if len(ys) > len(xs):
      with dep(ys[len(xs):]):
        return [tf.identity(x) for x in [incr(i), *ys[0:len(xs)]]]
    else:
      return [incr(i), *ys]
  return tf.while_loop(
      while_cond,
      while_body,
      [initial] + list(args),
      parallel_iterations=parallel_iterations,
      **kws)

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

def empty_page():
  return tf.constant(b'\xfe' * PGSIZE)

def has_page(addr):
  addr = ti64(addr)
  addr = PGROUNDDOWN(addr)
  o = optional_opt(G.mem.lookup(addr), tf.string)
  ok = o.has_value()
  return ok

def shapeof(x):
  if hasattr(x, 'shape'):
    x = x.shape
  if hasattr(x, 'as_list'):
    x = x.as_list()
  return x

def dim(x):
  return len(shapeof(x))

def cast_page(page, dtype, set_shape=True):
  page = tf.convert_to_tensor(page)
  if page.dtype == dtype:
    return page
  if page.dtype != tf.string:
    raise NotImplementedError()
  if dtype != tf.string:
    page = tf.io.decode_raw(page, dtype)
    if set_shape:
      shape = [None] * dim(page)
      shape[-1] = PGSIZE // dtype.size
      page.set_shape(shape)
      # if dtype != tf.uint8:
      #   page = tf.bitcast(page, dtype)
  return page

def join_pages(pages):
  pages = tf.convert_to_tensor(pages)
  if dim(pages) <= 0:
    return pages
  if pages.dtype == tf.string:
    return tf.strings.reduce_join(pages)
  elif dim(pages) > 1:
    return tf.reshape(pages, [-1])

def get_page(addr, dtype=tf.string):
  addr = ti64(addr)
  addr = PGROUNDDOWN(addr)
  o = optional_opt(G.mem.lookup(addr), tf.string)
  ok = o.has_value()
  page = tf.cond(ok, lambda: o.get_value(), lambda: empty_page())
  page = cast_page(page, dtype)
  return ok, page

# fall back to while loops for pfor
from tensorflow.python.ops.parallel_for.pfor import flags as pfor_flags
pfor_flags.FLAGS.op_conversion_fallback_to_while_loop = True

def pageof(addr, dtype=tf.string, pagefault=True):
  def inner(addr):
    ok, page = get_page(addr, dtype)
    if pagefault:
      page = check(page, ok, "Pagefault when reading",
          "addr=", addr)
    return page
  addr = ti64(addr)
  if addr.shape.as_list() == []:
    return inner(addr)
  else:
    pages = tf.map_fn(inner, addr, dtype=dtype, parallel_iterations=96)
    # if dtype != tf.string:
    #   pages.set_shape([None, PGSIZE])
    return pages

def ispage(addr, dtype=tf.int64, unset=0):
  def inner(addr):
    nonlocal unset
    if callable(unset):
      unset = unset(addr)
    return tf.where(has_page(addr), addr, unset)
  addr = ti64(addr)
  if addr.shape.as_list() == []:
    return inner(addr)
  else:
    result = tf.map_fn(inner, addr, dtype=dtype, parallel_iterations=96)
    return result

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
  ta = api.tf_array_new(tf.uint8, dynamic_size=False, size=tf.size(buf))
  ta = api.tf_array_extend(ta, buf, start=0)
  return ta

def memset_slow(p, byte, size):
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

def memcpy_slow(dst, src, size):
  dst = ti64(dst)
  size = ti64(size)
  dst_end = dst + size
  src = tf.convert_to_tensor(src)
  src = pagebuf(src)
  def body(addr, end, page):
    if page is None:
      page = empty_page()
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

def pageaddr(dst, size=1):
  dst = ti64(dst)
  pg_start = PGROUNDDOWN(dst)
  pg_end = PGROUNDUP(dst + size)
  #pg_step = tf.reshape(ti64(PGSIZE), dst.shape)
  pg_step = ti64(PGSIZE)
  #addrs = tf.unique( PGROUNDDOWN( tf.range(dst, dst_end, 1) ) )[0]
  addrs = tf.range(pg_start, pg_end, pg_step, dtype=tf.int64)
  return addrs

def memput(addr, value, size, dtype=None):
  src = tf_io_encode_raw(value, dtype=dtype)
  return memcpy(addr, src, size)

def memset(dst, value, size):
  value = tf.convert_to_tensor(value, tf.uint8)
  data = tf.fill([size], value)
  return memput(dst, data, size)

def memcpy(dst, src, size):
  dst = ti64(dst)
  size = ti64(size)
  beg = PGROUNDDOWN(dst)
  off = dst - beg
  end = dst + size
  upto = end - PGROUNDDOWN(end) - PGSIZE
  data = memof(dst, size, tf.string, pagefault=False, clip=False)
  lh = cut(data, 0, off)
  md = tf.convert_to_tensor(src)
  rh = cut(data, upto)
  data = join_pages([lh, md, rh])
  data = cast_page(data, tf.uint8, set_shape=False)
  data = tf.reshape(data, [-1, PGSIZE])
  pages = tf.map_fn(lambda x: optional_ref( tf_io_encode_raw(x) ), data, tf.variant)
  addrs = pageaddr(dst, size)
  return G.mem.insert(addrs, pages)

def panic(condition, msg, *args):
  return tf.debugging.Assert(condition, [msg, *args])

def check(x, condition, msg, *args):
  if callable(condition):
    condition = condition(x)
  with dep(panic(condition, msg, *args)):
    return tf.identity(x)

def memread_slow(dst, size, dtype=tf.string, pagefault=True):
  dst = ti64(dst)
  size = ti64(size)
  dst_end = dst + size
  ta = api.tf_array_new(tf.uint8)
  def body(addr, end, page, ta):
    if page is None:
      page = tf.constant(b'\xfe' * PGSIZE)
      if pagefault:
        page = check(page, False, "Pagefault when reading",
            "dst=", dst,
            "size=", size,
            "addr=", addr,
            "end=", end)
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


def cut(x, start=None, upto=None):
  x = tf.convert_to_tensor(x)
  n = api.tf_len(x)
  if start is None:
    start = ti64(0)
  if upto is None:
    upto = n
  start = tf.where(start < 0, start + n, start)
  upto = tf.where(upto < 0, upto + n, upto)
  upto = tf.where(upto < start, start, upto)
  size = upto - start
  if x.dtype == tf.string:
    x = tf.strings.substr(x, start, size)
  else:
    x = x[start:start+size]
  return x


def memof(dst, size, dtype=tf.string, pagefault=True, clip=False):
  dst = ti64(dst)
  size = ti64(size)
  # size = tf.reshape(size, dst.shape)
  # pg_step = tf.reshape(ti64(PGSIZE), dst.shape)
  pg_step = ti64(PGSIZE)
  dst_end = dst + size
  pg_start = PGROUNDDOWN(dst)
  pg_end = PGROUNDUP(dst_end)
  #addrs = tf.unique( PGROUNDDOWN( tf.range(dst, dst_end, 1) ) )[0]
  addrs = tf.range(pg_start, pg_end, pg_step, dtype=tf.int64)
  out = pageof( addrs, dtype, pagefault=pagefault )
  out = join_pages(out)
  if clip:
    off = dst - pg_start
    if dtype != tf.string:
      off //= dtype.size
      size //= dtype.size
    out = cut(out, off, off+size)
  return out

def memread(dst, size, dtype=tf.string, pagefault=True):
  out = memof(dst, size, dtype=tf.string, pagefault=pagefault, clip=True)
  if dtype != tf.string:
    out = tf.io.decode_raw(out, dtype)
    if isinstance(size, int):
      shape = [None] * dim(out)
      shape[-1] = size // dtype.size
      out.set_shape(shape)
  return out
    
def strlen_slow(src):
  src = ti64(src)
  dst = loop(src,
      lambda i: get_char(i)[0],
      lambda x: tf.math.not_equal(x, 0),
      lambda x: x)
  return dst - src

# clean this up...
def strlen(src):
  def chk(x):
    return tf.where(
        tf.reduce_any(tf.math.equal(x, 0)),
        tf.argmin(x),
        -1)
  src = ti64(src)
  end = PGROUNDUP(PGROUNDDOWN(src)+1)
  head = memread(src, end - src, tf.uint8)
  idx = chk(head)
  def rest():
    i, j = tf.while_loop(
        lambda i, j: j < 0,
        lambda i, j: (i + PGSIZE, chk(pageof(i, tf.uint8))),
        (end, ti64(-1)),
        )
    return (ti64(i) - PGSIZE + ti64(j)) - src
  return tf.cond(idx >= 0,
      lambda: idx,
      lambda: rest())

  
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


from copy import copy

def get_var_list(existing=None):
  if existing is None:
    existing = []
  existing = copy(existing)
  existing += [tf.lookup.experimental.DenseHashTable._Saveable( G.mem, 'mem' )]
  existing += [tf.lookup.experimental.DenseHashTable._Saveable( G.ptr, 'ptr' )]
  return existing

# tf.train.Saver(var_list=[tf.lookup.experimental.DenseHashTable._Saveable( xv6.G.mem, 'mem' )]).restore(sess, 'gs://ml-euw4/tmp/feb11/xv6_mem')

class Saver(tf.train.Saver):
  def __init__(self, var_list=None, path='gs://ml-euw4/tmp/feb12/xv6', **kws):
    self.path = path
    self.var_list = get_var_list(var_list)
    super().__init__(self.var_list, **kws)

  def save(self, session=None, path=None, write_meta_graph=False, **kws):
    if path is None:
      path = self.path
    if session is None:
      session = tf.get_default_session()
    return super().save(session, path, write_meta_graph=write_meta_graph, **kws)

  def restore(self, session=None, path=None, **kws):
    if path is None:
      path = self.path
    if session is None:
      session = tf.get_default_session()
    return super().restore(session, path, **kws)


import _ctypes
import ctypes
ct = ctypes
#from ctypes import *

class EmxArray(ct.Structure):
    """ creates a struct to match emxArray_real_T """
    _fields_ = [('data', ct.POINTER(ct.c_double)),
                ('size', ct.POINTER(ct.c_int)),
                ('allocatedSize', ct.c_int),
                ('numDimensions', ct.c_int),
                ('canFreeData', ct.c_bool)]

#class _print_fields(type(ct.Structure)):
class _print_fields(type(ct.Structure)):
    def __getattr__(self, attrname, *args, **kws):
        print('{}.__getattr__'.format(self), attrname, args, kws)
        return super().__getattr__(attrname, *args, **kws)
    def __setattr__(self, attrname, value):
        print('{}.__setattr__'.format(self), attrname, value)
        # if attrname == "_fields_":
        #     fields = []
        #     for desc in value:
        #         name = desc[0]
        #         typ = desc[1]
        #         rest = desc[2:]
        #         fields.append((name, _other_endian(typ)) + rest)
        #     value = fields
        super().__setattr__(attrname, value)

class Packet(ct.Structure, metaclass=_print_fields):
    _fields_ = [("id", ct.c_int),
                ("ce", ct.POINTER(ct.c_ubyte)),
                ("syms", ct.POINTER(ct.c_ubyte))]

class Blob(ct.Structure, metaclass=_print_fields):
    _fields_ = [("size", ct.c_uint64),
                ("data", (ct.c_ubyte * 0)),
                ]

class _RemoteData:
    def __init__(self, address=None):
      super().__init__()
      self.address = address
    
    @property
    def b_size(self):
      return ct.sizeof( self.__class__._type_ )
        
    @classmethod
    def from_address(cls, addr):
      print('from_address', cls, addr)
      return cls(addr)

class _RemoteMC(type):
    def __new__(cls, name, bases, dct):
        bases = tuple([_RemoteData] + list(bases))
        x = super().__new__(cls, name, bases, dct)
        print('{}.__new__'.format(x), cls, name, bases, dct)
        assert hasattr(x, '_type_'), "Set _type_ = <some cdata structure>"
        return x
    def __getattr__(self, attrname, *args, **kws):
        print('{}.__getattr__'.format(self), attrname, args, kws)
        return super().__getattr__(attrname, *args, **kws)
    def __setattr__(self, attrname, value):
        print('{}.__setattr__'.format(self), attrname, value)
        # if attrname == "_fields_":
        #     fields = []
        #     for desc in value:
        #         name = desc[0]
        #         typ = desc[1]
        #         rest = desc[2:]
        #         fields.append((name, _other_endian(typ)) + rest)
        #     value = fields
        super().__setattr__(attrname, value)

class RemoteBlob(metaclass=_RemoteMC):
    _type_ = Blob

class RemoteU64(metaclass=_RemoteMC):
    _type_ = ct.c_uint64

def serialize(pkt_p, size_g, size_p):
    """ Serialize Packet instance
        size_g - number of elements pointed by ce
        size_p - number of elements pointed by syms
        Return a byte stream
    """ 
    pktstr = b''
    pktstr += struct.pack('i', pkt_p.contents.id)
    pktstr += string_at(pkt_p.contents.ce, size_g)
    pktstr += string_at(pkt_p.contents.syms, size_p)
    return pktstr
    

class Ptr(ct.POINTER(Packet)):
  _type_ = Packet

  @property
  def contents(self):
    return tf.no_op()

def to_char_array(p):
  pdata = ct.cast(ct.byref(p), ct.POINTER(ct.c_char * ct.sizeof(p)))



_array_type = type(ct.Array)

def _other_endian(typ):
    """Return the type with the 'other' byte order.  Simple types like
    c_int and so on already have __ctype_be__ and __ctype_le__
    attributes which contain the types, for more complicated types
    arrays and structures are supported.
    """
    # check _OTHER_ENDIAN attribute (present if typ is primitive type)
    if hasattr(typ, _OTHER_ENDIAN):
        return getattr(typ, _OTHER_ENDIAN)
    # if typ is array
    if isinstance(typ, _array_type):
        return _other_endian(typ._type_) * typ._length_
    # if typ is structure
    if issubclass(typ, ct.Structure):
        return typ
    raise TypeError("This type does not support other endian: %s" % typ)

class _swapped_meta(type(ct.Structure)):
    def __setattr__(self, attrname, value):
        if attrname == "_fields_":
            fields = []
            for desc in value:
                name = desc[0]
                typ = desc[1]
                rest = desc[2:]
                fields.append((name, _other_endian(typ)) + rest)
            value = fields
        super().__setattr__(attrname, value)



get_long = functools.partial(memread, size=4, dtype=tf.int32)
get_ulong = functools.partial(memread, size=4, dtype=tf.uint32)
get_longlong = functools.partial(memread, size=8, dtype=tf.int64)
get_ulonglong = functools.partial(memread, size=8, dtype=tf.uint64)
get_double = functools.partial(memread, size=8, dtype=tf.float64)
get_float = functools.partial(memread, size=4, dtype=tf.float32)
get_char = functools.partial(memread, size=1, dtype=tf.uint8)

set_long = functools.partial(memput, size=4, dtype=tf.int32)
set_ulong = functools.partial(memput, size=4, dtype=tf.uint32)
set_longlong = functools.partial(memput, size=8, dtype=tf.int64)
set_ulonglong = functools.partial(memput, size=8, dtype=tf.int64)
set_double = functools.partial(memput, size=8, dtype=tf.float64)
set_float = functools.partial(memput, size=4, dtype=tf.float32)
set_char = functools.partial(memput, size=1, dtype=tf.uint8)


def set_blob(addr, value, size=None):
  if size is None:
    size = api.tf_len(value)
  addr = ti64(addr)
  size = ti64(size)
  # WARNING: this won't work! either the first op or the second op
  # will apply, but not both, due to pointer aliasing.
  # return [
  #     set_longlong(addr, size),
  #     memcpy(addr + 8, value, size),
  #     ]
  with dep(set_longlong(addr, size)):
    with dep(memcpy(addr + 8, value, size)):
      return addr + 8 + size

def get_blob(addr):
  addr = ti64(addr)
  size = get_longlong(addr)[0]
  return memread(addr + 8, size)

def get_blob_end(addr, n=1):
  addr = ti64(addr)
  def inner(i, addr):
    size = get_longlong(addr)[0]
    return addr + 8 + size
  _, addr = fori(0, n, 1, inner, addr)
  return addr

# addr_ph = tf.placeholder(tf.int64, shape=()); data_ph = tf.placeholder(tf.string, shape=())
## op = [xv6.get_blob_end(addr_ph, 2), tf.shape(tf.io.decode_image(xv6.get_blob(xv6.get_blob_end(addr_ph,1)), channels=3))]
# op = [xv6.get_blob_end(addr_ph, 2), xv6.get_blob(addr_ph), tf.shape(tf.io.decode_image(xv6.get_blob(xv6.get_blob_end(addr_ph,1)), channels=3))]
# pt = xv6.KERNBASE
# >>> pt, name, shape = r( op, {addr_ph: pt} ); list(shape), name
# ([423, 634, 3], b'/Volumes/birdie/data/conceptual_captions/download/https/i.dailymail.co.uk/i/pix/2012/11/20/article-2235635-162052EB000005DC-161_634x423.jpg')
# >>> pt, name, shape = r( op, {addr_ph: pt} ); list(shape), name
# ([954, 634, 3], b'/Volumes/birdie/data/conceptual_captions/download/https/i.dailymail.co.uk/i/pix/2012/10/05/article-0-155E3D30000005DC-463_634x954.jpg')
# >>> pt, name, shape = r( op, {addr_ph: pt} ); list(shape), name
# ([472, 634, 3], b'/Volumes/birdie/data/conceptual_captions/download/https/i.dailymail.co.uk/i/pix/2012/10/02/article-0-15460B26000005DC-969_634x472.jpg')
# >>> pt, name, shape = r( op, {addr_ph: pt} ); list(shape), name
# ([738, 634, 3], b'/Volumes/birdie/data/conceptual_captions/download/https/i.dailymail.co.uk/i/pix/2012/05/07/article-2140693-12F69005000005DC-866_634x738.jpg')
# >>> pt, name, shape = r( op, {addr_ph: pt} ); list(shape), name
# ([924, 634, 3], b'/Volumes/birdie/data/conceptual_captions/download/https/i.dailymail.co.uk/i/pix/2012/02/01/article-2094529-118B44F7000005DC-452_634x924.jpg')
# >>> pt, name, shape = r( op, {addr_ph: pt} ); list(shape), name
# ([817, 634, 3], b'/Volumes/birdie/data/conceptual_captions/download/https/i.dailymail.co.uk/i/pix/2012/12/16/article-2248872-168BD299000005DC-417_634x817.jpg')
# >>> pt, name, shape = r( op, {addr_ph: pt} ); list(shape), name
# ([466, 634, 3], b'/Volumes/birdie/data/conceptual_captions/download/https/i.dailymail.co.uk/i/pix/2012/03/31/article-2123302-12694D1A000005DC-581_634x466.jpg')
# >>> pt, name, shape = r( op, {addr_ph: pt} ); list(shape), name
# ([444, 638, 3], b'/Volumes/birdie/data/conceptual_captions/download/https/i.dailymail.co.uk/i/pix/2013/05/17/article-0-19D3BEED000005DC-312_638x444.jpg')
# >>> pt, name, shape = r( op, {addr_ph: pt} ); list(shape), name
# ([478, 634, 3], b'/Volumes/birdie/data/conceptual_captions/download/https/i.dailymail.co.uk/i/pix/2013/01/28/article-2269327-17347F73000005DC-638_634x478.jpg')
# >>> pt, name, shape = r( op, {addr_ph: pt} ); list(shape), name
# ([455, 634, 3], b'/Volumes/birdie/data/conceptual_captions/download/https/i.dailymail.co.uk/i/pix/2013/10/06/article-2446494-188E3ACA00000578-613_634x455.jpg')
