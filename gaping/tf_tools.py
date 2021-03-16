import tensorflow as tf
tf1 = tf.compat.v1
import numpy as np
import sys
import os

from tensorflow.python.platform import tf_logging as logging

from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.tpu import tpu
from tensorflow.compiler.tf2xla.python import xla
from tensorflow.compiler.tf2xla.ops import gen_xla_ops
from tensorflow.python.tpu import tpu_strategy_util
from tensorflow.python.tpu import device_assignment as device_assignment_lib
from tensorflow.python.tpu import topology as topology_lib
from tensorflow.python.tpu import tpu as tpu_lib



def after(op, then, pass_arg=False):
  op = [op] if not isinstance(op, (list, tuple)) else op
  with tf.control_dependencies(op):
    #return tf.identity(then())
    if pass_arg:
      return then(op)
    else:
      return then()


def count_tpu_cores(session=None):
  if session is None:
    session = tf1.get_default_session()
  return len([x for x in session.list_devices() if ':TPU:' in x.name])

import functools

def tpu_shard(op, device_assignment=None, num_shards=None, outputs_from_all_shards=True, **kws):
  if num_shards is None:
    if device_assignment is not None:
      num_shards = len(device_assignment.core_assignment)
    else:
      num_shards = count_tpu_cores()
  return tpu_lib.shard(op, outputs_from_all_shards=outputs_from_all_shards, num_shards=num_shards, device_assignment=device_assignment, **kws)

def tpu_batch(op, inputs, device_assignment=None, num_shards=None, **kws):
  if num_shards is None:
    if device_assignment is not None:
      num_shards = len(device_assignment.core_assignment)
    else:
      num_shards = count_tpu_cores()
  return tpu_lib.batch_parallel(op, inputs=inputs, num_shards=num_shards, device_assignment=device_assignment, **kws)

def tpu_id():
  # TODO(iamtingchen): more elegant way to convert u32 to s32 for replica_id.
  replica_id = tf.cast(tf.cast(xla.replica_id(), tf.uint32), tf.int32)
  return replica_id

def get_outside_context(graph=None):
  if graph is None:
    graph = ops.get_default_graph()
  # pylint: disable=protected-access
  context = graph._get_control_flow_context()
  # pylint: enable=protected-access
  while context is not None and not hasattr(context, '_outside_compilation_counter'):
    context = context.outer_context
  return context

def tpu_cpu(f, *args, mutex=None, **kws):
  graph = tf.get_default_graph()
  context = get_outside_context(graph)
  #print(context._outside_compilation_cluster)
  #print(context._outside_compilation_counter)
  def inner(*args, **kws):
    if mutex is not None:
      return mutex.execute(lambda: f(*args, **kws))
    else:
      return f(*args, **kws)
  if context is not None and context._outside_compilation_counter > 0:
    return inner(*args, **kws)
  else:
    return tpu_lib.outside_compilation(inner, *args, **kws)

def tpu_now():
  return tpu_cpu(lambda: tf.identity(tf.timestamp(), name="timestamp"))

def tpu_nanos():
  @tpu_cpu
  def on_cpu():
    now = tf.timestamp()
    rem = tf.math.floormod(now, 1.0)
    now = tf.cast(now, tf.int64) * (10**9) + \
          tf.cast(rem * 1e9, tf.int64)
    return now
  return on_cpu


def tpu_thunk(i, label, image_bytes):
  print(label)
  print(image_bytes)
  def on_cpu(imgbytes):
    tf.io.decode_image(x,)
  tf.add(i,label)

def spread(fn): return lambda args: fn(*args)


def map_fn(dtypes, xs, fn):
  return tf.map_fn(lambda args: fn(*args), xs, infer_shape=False, dtype=dtypes)#[x.dtype for x in xs])
# iz0 = tft.map_fn(de, lambda label, img: [label, tft.transform_image(img, [128, 128, 3])])
# iz = tf.vectorized_map(lambda x: tf.io.decode_image(x, channels=3, dtype=tf.float32), de)
# iz2 = tf.map_fn(lambda x: [x[0], tf.io.decode_image(x[1], channels=3)], de, infer_shape=False, dtype=[tf.int32, tf.uint8])
# iz6 = tf.map_fn(lambda x: [x[0], tft.transform_image(tft.transform_image(x[1], target_image_shape=[128, 64, 3], crop_method='resize_with_pad'), crop_method='top')], de, infer_shape=False, dtype=[tf.int32, tf.float32])
# iz6 = tf.map_fn(lambda x: [x[0], tft.transform_image(tft.transform_image(x[1], target_image_shape=[128, 64, 3], crop_method='resize_with_pad'), crop_method='top')], de, infer_shape=False, dtype=[tf.int32, tf.float32])

def transform_image(image, target_image_shape=None, crop_method='none', seed=None, resize_method=tf.image.ResizeMethod.AREA, channels=3, dtype=tf.float32, align_corners=True):
  """Preprocesses ImageNet images to have a target image shape.

  Args:
    image: 3-D tensor with a single image.
    target_image_shape: List/Tuple with target image shape.
    crop_method: Method for cropping the image:
      One of: distorted, random, middle, none
    seed: Random seed, only used for `crop_method=distorted`.

  Returns:
    Image tensor with shape `target_image_shape`.
  """
  print('TKTK', image)
  if image.dtype == tf.string:
    image = tf.io.decode_image(image, channels=channels, dtype=dtype)
  shape = tf.shape(image)
  print('TKTK', image, shape)
  image = tf.image.convert_image_dtype(image, dtype)
  if crop_method == "distorted":
    begin, size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        tf.zeros([0, 0, 4], tf.float32),
        aspect_ratio_range=[1.0, 1.0],
        area_range=[0.9, 1.0],
        use_image_if_no_bounding_boxes=True,
        seed=seed)
    image = tf.slice(image, begin, size)
    # Unfortunately, the above operation loses the depth-dimension. So we need
    # to restore it the manual way.
    #image.set_shape([None, None, target_image_shape[-1]])
  elif crop_method == "random":
    tf.set_random_seed(seed)
    shape = tf.shape(image)
    h, w = shape[0], shape[1]
    size = tf.minimum(h, w)
    begin = [h - size, w - size] * tf.random.uniform([2], 0, 1)
    begin = tf.cast(begin, tf.int32)
    begin = tf.concat([begin, [0]], axis=0)  # Add channel dimension.
    image = tf.slice(image, begin, [size, size, 3])
  elif crop_method == "middle":
    shape = tf.shape(image)
    h, w = shape[0], shape[1]
    size = tf.minimum(h, w)
    begin = tf.cast([h - size, w - size], tf.float32) / 2.0
    begin = tf.cast(begin, tf.int32)
    begin = tf.concat([begin, [0]], axis=0)  # Add channel dimension.
    image = tf.slice(image, begin, [size, size, 3])
  elif crop_method == "top":
    shape = tf.shape(image)
    h, w = shape[0], shape[1]
    size = tf.minimum(h, w)
    begin = tf.cast([0, w - size], tf.float32) / 2.0
    begin = tf.cast(begin, tf.int32)
    begin = tf.concat([begin, [0]], axis=0)  # Add channel dimension.
    image = tf.slice(image, begin, [size, size, 3])
  elif crop_method == "resize_with_pad":
    image = tf.image.resize_image_with_pad(
      image, target_image_shape[0], target_image_shape[1],
      method=resize_method,
      align_corners=align_corners)
    image.set_shape(target_image_shape)
    return image
  elif crop_method != "none":
    raise ValueError("Unsupported crop method: {}".format(crop_method))
  if target_image_shape is not None:
    image = tf.image.resize_images(
        image, [target_image_shape[0], target_image_shape[1]],
        method=resize_method)
    image.set_shape(target_image_shape)
  return image

def save_image_grid(x, fname=None, normalize=False, pad_value=0.5**(1/2.2), return_grid=False, **kws):
  if np.ndim(x) == 3:
     x = np.expand_dims(x, 0)
  x = np.transpose(x, [0,3,1,2])
  from torchvision.utils import save_image
  from torchvision.utils import make_grid
  import torch
  t = torch.tensor(x)
  if fname:
    save_image(t, fname, normalize=normalize, pad_value=pad_value, **kws)
  if return_grid:
    t = make_grid(t, normalize=normalize, pad_value=pad_value, **kws)
    x = t.numpy()
    if np.ndim(x) == 3:
      x = np.expand_dims(x, 0)
    x = np.transpose(x, [0,2,3,1])
    return x
  
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

# https://stackoverflow.com/questions/45179820/draw-text-on-an-angle-rotated-in-python

def tensor_to_pil(tensor):
  if isinstance(tensor, Image.Image):
    return tensor
  if isinstance(tensor, (tuple, list)) and all([isinstance(x, Image.Image) for x in tensor]):
    return tensor
  if np.ndim(tensor) == 4:
    return [tensor_to_pil(x) for x in tensor]
  if tensor.dtype in ['float32', 'float64']:
    tensor = np.clip(tensor * 255.0, 0, 255).astype(np.uint8)
  return Image.fromarray(tensor)

from io import BytesIO

def bytes_to_pil(data):
  return Image.open(BytesIO(data))

def pil_to_bytes(image, quality=60):
  with BytesIO() as bio:
    image = image.convert('RGB')
    image.save(bio, 'JPEG', quality=quality)
    return bio.getvalue()

def pil_to_tensor(image):
  image = np.array(image, np.uint8) / 255.0
  return image

def label_image_grid(labels, images):
  r = []
  for label, image in zip(labels, images):
    image = tensor_to_pil(image)
    draw = ImageDraw.Draw(image)
    draw.text((2, 2), str(label), (255, 255, 255))
    image = pil_to_tensor(image)
    r.append(image)
  return r

# char_image = np.zeros((200, 300, 3), np.uint8)

# # convert to pillow image
# pillowImage = Image.fromarray(char_image)
# draw = ImageDraw.Draw(pillowImage)

# # add chars to image
# font = ImageFont.truetype("arial.ttf", 32)
# draw.text((50, 50), 'ABC', (255, 255, 255), font=font)

# # convert back to numpy array
# char_image = np.array(pillowImage, np.uint8)

def readbytes(filename):
  with tf.io.gfile.GFile(filename, 'rb') as f:
    return f.read()

def writebytes(filename, data):
  with tf.io.gfile.GFile(filename, 'wb') as f:
    return f.write(data)


from tensorflow.python.tpu import tpu_function

def replica_count():
  return tpu_function.get_tpu_context().number_of_shards or 1


def tpu_concat(tensor):
  """Reduce a concatenation of the `tensor` across TPU cores.

  Args:
    tensor: tensor to concatenate.

  Returns:
    Tensor of the same rank as `tensor` with first dimension `num_replicas`
    times larger.
  """
  num_replicas = replica_count()
  replica_id = tpu_id()

  with tf.name_scope('tpu_cross_replica_concat'):
    # This creates a tensor that is like the input tensor but has an added
    # replica dimension as the outermost dimension. On each replica it will
    # contain the local values and zeros for all other values that need to be
    # fetched from other replicas.
    ext_tensor = tf.scatter_nd(
        indices=[[replica_id]],
        updates=[tensor],
        shape=[num_replicas] + tensor.shape.as_list())

    # As every value is only present on one replica and 0 in all others, adding
    # them all together will result in the full tensor on all replicas.
    ext_tensor = tf.tpu.cross_replica_sum(ext_tensor)

    return ext_tensor
    # # Flatten the replica dimension.
    # # The first dimension size will be: tensor.shape[0] * num_replicas
    # # Using [-1] trick to support also scalar input.
    # return tf.reshape(ext_tensor, [-1] + ext_tensor.shape.as_list()[2:])


def with_device(name, thunk):
  if hasattr(name, 'name'):
    name = name.name
  with tf.device(name):
    return thunk()

def with_shape(shape, thunk):
  result = thunk()
  result.set_shape(shape)
  return result


from tensorflow.python.eager import def_function

# def_function.function(func=None, input_signature=None, autograph=True, experimental_autograph_options=None, experimental_relax_shapes=False, experimental_compile=None)

def defun(fn, compile=True, **kws):
  # fix the following case:
  #
  #   op = tf.raw_ops.TPUPartitionedCall(args=[], device_ordinal=tf.random.uniform([],minval=8, maxval=16, dtype=tf.int32), Tout=[tf.int32], f=tft.defun(lambda: tf.add(0,tft.tpu_id())).get_concrete_function())
  #
  # which results in an error: Node '__inference_<lambda>_5267_ord_11_0/XlaReplicaId': Node name contains invalid characters
  #
  fn.__name__ = fn.__name__.replace('<lambda>', '__lambda__')
  return def_function.function(fn, experimental_compile=compile, **kws)


from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops


def enclosing_tpu_context():
  # pylint: disable=protected-access
  context = ops.get_default_graph()._get_control_flow_context()
  # pylint: enable=protected-access
  while context is not None and not isinstance(
      context, control_flow_ops.XLAControlFlowContext):
    context = context.outer_context
  return context


def set(name, value, globals=sys.modules['__main__'].__dict__):
  globals[name] = value
  return value


from tensorflow.python.framework import indexed_slices

# learned this trick from _colocate_with_for_gradient in tensorflow/tensorflow/python/framework/ops.py
as_tensor = indexed_slices.internal_convert_to_tensor_or_indexed_slices
as_list = indexed_slices.internal_convert_n_to_tensor_or_indexed_slices


class FakeOp(object):
  """A helper class to determine the current device.

  Supports only the type and device set/get methods needed to run the
  graph's _apply_device_function method.
  """

  def __init__(self):
    self._device = ""

  @property
  def type(self):
    return "FakeOp"

  @property
  def device(self):
    return self._device

  def _set_device(self, device):
    if isinstance(device, pydev.DeviceSpec):
      self._device = device.to_string()
    else:
      self._device = device

  def _set_device_from_string(self, device_str):
    self._device = device_str

from tensorflow.python.framework import device as pydev

def current_device_name():
  graph = ops.get_default_graph()
  fake_op = FakeOp()
  graph._apply_device_functions(fake_op)  # pylint: disable=protected-access
  device = pydev.DeviceSpec.from_string(fake_op.device)
  return device.to_string()


from tensorflow.python.framework import device_spec
from collections import OrderedDict

def normalize_device_name(name=None, *, toplevel=True, **kws):
  if toplevel:
    if len(kws) > 0:
      name = (name or '') + ' ' + '/'.join([':'.join([k, str(v)]) for k, v in kws.items()])
    if name is None:
      return name
    name = name.replace('=', ':')
    name = name.replace(' ', '/')
    name = name.replace(',', '/')
    name = name.strip('/')
    if 'device:' not in name:
      name = 'device:' + name
    parts = [normalize_device_name(x, toplevel=False) for x in name.split('/')]
    parts = [x for x in parts if x]
    props = OrderedDict([x.split(':', 1) for x in parts])
    # this is probably a bad idea; skip for now.
    # if 'device' in props:
    #   if ':' in props['device']:
    #     kind, core = props['device'].rsplit(':', 1)
    #     if core == '*':
    #       core = '0'
    #     props['core'] = str(int(props.pop('core', '0')) + int(core))
    #     props['device'] = kind
    # if 'device' in props:
    #   task = int(props.pop('task', '0'))
    #   if 'core' in props and props['core'] != '*':
    #     task = 0
    #   core = props.pop('core', '*')
    #   if core == '*':
    #     core = 0
    #   else:
    #     core = int(core)
    #   cores_per_task = 8 if props['device'].startswith('TPU') else 1
    #   print('CORE', core, task)
    #   # while task > 0:
    #   #   core += cores_per_task
    #   #   task -= 1
    #   print('CORE', core, task)
    #   while core >= cores_per_task:
    #     task += 1
    #     core -= cores_per_task
    #   props['task'] = str(task)
    #   props['device'] = props.pop('device').split(':', 1)[0] + ':' + str(core)
    final = '/' + '/'.join([':'.join([k, v]) for k, v in props.items()])
    return final
  if not name:
    return name
  if ':' not in name:
    name = 'device:{}'.format(name.upper())
  name, kind, *value = name.split(':')
  if name == 'device' and not value:
    value += ['*']
  if name.upper() in ['CPU', 'GPU', 'TPU', 'TPU_SYSTEM']:
    name, kind, value = 'device', name, [kind] + value
  if name == 'device':
    kind = kind.upper()
  if kind == 'XLA':
    kind = 'XLA_CPU'
  if kind == 'CORE':
    kind = 'TPU_REPLICATED_CORE'
  assert not (name == 'job' and kind == 'master'), "I kept typing job=master instead of job=worker, so this is a personal sanity check"
  return ':'.join([name, kind, *value])

def parse_device_name(name=None, **kws):
  if name is None:
    name = current_device_name()
  if not isinstance(name, str):
    # assume it's a DeviceSpec
    name = name.to_string()
  name = normalize_device_name(name, **kws)
  spec = device_spec.DeviceSpecV2()
  spec = spec.parse_from_string(name)
  return spec

def device_name(name=None, **kws):
  return parse_device_name(name, **kws).to_string()

def device(name=None, **kws):
  return tf.device(device_name(name, **kws))

def with_device(name, thunk):
  with device(name):
    return thunk()


import contextlib

# from ./mesh_tensorflow/utils.py

@contextlib.contextmanager
def outside_all_rewrites():
  with ops.control_dependencies(None):
    yield
