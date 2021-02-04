import tensorflow as tf

from .. import tf_api
from ..data import queue
from .. import tf_tools as tft

#q3op = queue.queue_tpu(q3, lambda fname, fdata: [1, tf.reduce_prod(tft.tpu_cpu(lambda x: tf.shape(tf.io.decode_image(x[0], channels=3, dtype=tf.float32)), fdata))])

def q3_thunk(fname, fdata, size):
  def on_cpu(fname, fdata):
    n = tf.size(fdata)
    n = tf.guarantee_const(n)
    def loop(i):
      x = fdata[i]
      image = tf.io.decode_image(x, channels=3, dtype=tf.float32)
      shape = tf.shape(image)
      shape.set_shape([3])
      H, W, C = tf.unstack(shape, 3 )
      image = tf.pad(image, [[2048-H, 0], [2048 - W, 0], [0, 0]], constant_values=0.0)
      image.set_shape([2048, 2048, 3])
      print(image.shape)
      print(shape, H, W, C)
      return image
    def loop_2(i):
      x = fdata[i]
      image = tf.io.decode_image(x, channels=3, dtype=tf.int32)
      shape = tf.shape(image)
      shape.set_shape([3])
      H, W, C = tf.unstack(shape, 3 )
      image = tf.pad(image, [[2048-H, 0], [2048 - W, 0], [0, 0]], constant_values=0.0)
      image.set_shape([2048, 2048, 3])
      print(image.shape)
      print(shape, H, W, C)
      return shape
    #fdata.set_shape([n])
    #result = tf.vectorized_map(loop, tf.range(n))
    #print(n)
    #return loop(fdata[0])
    #return result
    ta = tf_api.tf_array_new(tf.int32)
    ta = tf_api.tf_array_extend(ta, tf.range(n))
    ta_image = ta
    ta = tf_api.tf_array_new(tf.int32)
    ta = tf_api.tf_array_extend(ta, tf.range(n))
    ta_shape = ta
    ta_image = tf_api.tf_array_map(ta_image, loop, out_dtype=tf.float32)
    ta_shape = tf_api.tf_array_map(ta_shape, loop_2, out_dtype=tf.int32)
    #return [loop(0), loop(1)], n
    shapes = ta_shape.stack()
    images = ta_image.stack()
    #import pdb; pdb.set_trace()
    shapes = tf_api.tf_set_shape(shapes, [size, 3])
    images = tf_api.tf_set_shape(images, [size, 2048, 2048, 3])
    return shapes, images, n
  #[shape, shape2], n = tft.tpu_cpu(on_cpu, fname, fdata)
  shapes, images, n = tft.tpu_cpu(on_cpu, fname, fdata)
  print(shapes, images, n)
  #return tf.reduce_prod(shape), tf.reduce_prod(shape2), n
  #return n
  #return tf.reshape(tf.reduce_prod(shapes, axis=1), [-1, size]), n
  def inner(i):
    #return shape, image
    shape = shapes[i]
    image = images[i]
    return shape, image
  #shapes, images = tf.vectorized_map(inner, tf.range(size))
  #return tf.reshape(tf.reduce_prod(shapes, axis=1), [-1, size]), n
  shape, image = inner(0)
  #return tf.reshape(tf.reduce_prod(shape, axis=-1), [-1, size]), n
  H, W, C = tf.unstack(shape)
  #image = image[0:H, 0:W, 0:C]
  #return tf.reduce_prod(shape, axis=-1), tf.reduce_sum(image), n
  return fname, shapes


#op = queue.queue_tpu(q3, q3_thunk)


#q3op = queue.queue_tpu(q3, lambda fname, fdata: [1, tf.reduce_prod(tft.tpu_cpu(lambda x: tf.shape(tf.io.decode_image(x[0], channels=3, dtype=tf.float32)), fdata))])
