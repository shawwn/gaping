from . import biggan
from . import tftorch as nn

import numpy as np

from scipy.stats import truncnorm

from tensorflow.python.framework import ops
from tensorflow.python.tpu import tpu_feed

def truncated_z_sample(batch_size, z_dim, truncation=1.0, *, seed=None):
  if batch_size is None:
    return truncated_z_sample(1, z_dim=z_dim, truncation=truncation, seed=seed)[0]
  state = None if seed is None else np.random.RandomState(seed)
  values = truncnorm.rvs(-2, 2, size=(batch_size, z_dim), random_state=state)
  return (truncation * values).astype(np.float32)

def sample_labels(batch_size, num_classes, *, seed=None, hot=True):
  if batch_size is None:
    return sample_labels(1, num_classes=num_classes, seed=seed, hot=hot)[0]
  state = np.random.RandomState(seed=seed)
  labels = state.randint(0, num_classes, batch_size, dtype=np.int64)
  if hot:
    return one_hot(labels, num_classes)
  return labels

def one_hot(i, num_classes):
  if isinstance(i, int):
    i = [i]
    return one_hot(i, num_classes)[0]
  a = np.array(i, dtype=np.int32)
  #num_classes = a.max()+1
  b = np.zeros((a.size, num_classes))
  b[np.arange(a.size),a] = 1
  return b.astype(np.float32)


from tensorflow.python.tpu import tpu_feed
from tensorflow.python.tpu import tpu
from tensorflow_estimator.python.estimator.tpu import tpu_estimator
import tensorflow as tf
#rec = tpu_estimator._InputPipeline.InputsStructureRecorder( [ {'z': [1, 1], 'y': [1, 1]} , None ] )

# def read_blocks(next_block, count, dtype=tf.string, shape=[], *, parallel_iterations=1):
#   ta = tf.TensorArray(
#       size=0, element_shape=shape, dtype=dtype, dynamic_size=True)
#   _, filedata = tf.while_loop(
#       lambda *args: True,
#       lambda i, ta: (i + 1, ta.write(i, next_block())),
#       (0, ta),
#       maximum_iterations=count,
#       parallel_iterations=parallel_iterations)
#   return filedata.stack()

class InputQueue:
  def __init__(self, like, shared_name=None):
    self.structure = tf.nest.map_structure(tf.convert_to_tensor, like)
    flat = tf.nest.flatten(self.structure)
    names = tf.nest.flatten({k: k for k, v in self.structure.items()})
    dtypes = [x.dtype for x in flat]
    shapes = [x.shape for x in flat]
    self.queue = tf.FIFOQueue( 10_000_000, dtypes, shapes, names, shared_name=shared_name)

def device_for_tpu_core(task=0, core=0, job="worker"):
  #job_name = FLAGS.tpu_job_name or "tpu_worker"
  return "/job:%s/task:%d/device:TPU_REPLICATED_CORE:%d" % (job, task,
                                                            core)

def device_for_host(self, task=0, cpu=0, job="worker"):
  #job_name = FLAGS.tpu_job_name or "tpu_worker"
  return "/job:%s/task:%d/device:CPU:%d" % (job, task, cpu)



class BigGANSampler:
  def __init__(self, model, driver):
    self.model = model
    self.driver = driver
    if True:
      #self.device_assignment = driver.device_assignment()
      self.device_assignment = driver.device_assignment([list(range(8))])
    else:
      self.device_assignment = driver.device_assignment([[0,1,2,3], [4,5,6,7]])
      self.input = InputQueue( self.sample(4), shared_name="biggan_sampler_input" )
      #self.input_partition_dims = tf.nest.flatten({
      self.input_partition_dims = ({
        'z': [4],
        'y': [4],
        })

      # self.structure_recorder = tpu_estimator._InputPipeline.InputsStructureRecorder(self.input_partition_dims)
      # self.structure_recorder.validate_and_record_structure(self.sample(8), None)
      self.infeed_queue = tpu_feed._PartitionedInfeedQueue(
          2,
          self.device_assignment,
          0,
          input_partition_dims=self.input_partition_dims,
          # tuple_types=self.input.queue.dtypes,
          # tuple_shapes=self.input.queue.shapes,
          )
      # #self.infeed_queue.set_tuple_shapes(self.input.queue.shapes)
      # inputs = [self.input.queue.dequeue() for _ in range(self.device_assignment.num_replicas)]
      # flat_inputs = [
      #     tf.nest.flatten(per_replica_input) for per_replica_input in inputs
      # ]
      # # Converts inputs to Tensors.
      # flat_inputs = [[tf.convert_to_tensor(x) for x in inp] for inp in flat_inputs]
      # self.infeed_ops = self.infeed_queue.generate_enqueue_ops(flat_inputs)
      self.infeed_ops = []
      self.per_replica_inputs = []
      for replica in range(self.device_assignment.num_replicas):
        core = self.device_assignment.tpu_ordinal(replica=replica)
        per_replica_input = self.input.queue.dequeue()
        flat_input = tf.nest.flatten(per_replica_input)
        # infeed_op = self.infeed_queue.generate_enqueue_ops([flat_input])
        #self.flat_inputs.append(flat_input)
        self.per_replica_inputs.append(flat_input)
      inputs = [tf.stack(x) for x in list(zip(*self.per_replica_inputs))]
      #self.generate_op = self.driver.shard(self.generate, inputs=inputs, device_assignment=self.device_assignment)
    #self.generate_op = self.replicate(self.generate2)
    self.generate3_op = self.replicate(self.generate3)


  def replicate(self, fn, **kws):
    def inner(*args, **kws):
      op = fn(*args, **kws)
      with tf.device(device_for_tpu_core()):
        with tf.control_dependencies([op]):
          return op
    device_assignment = kws.pop('device_assignment', self.device_assignment)
    #return self.driver.shard(inner, device_assignment=device_assignment, **kws)
    compile_op, (results,) = tpu.split_compile_and_shard(
        inner,
        inputs=[],
        num_shards=self.device_assignment.num_replicas,
        outputs_from_all_shards=True,
        device_assignment=self.device_assignment,
        )
    return compile_op, results

  def generate2(self, generator=None, batch_size=1):
    if generator is None:
      generator = self.model.generator
    z = self.z_sample(batch_size=batch_size)
    y = self.y_sample(batch_size=batch_size)
    if True:
      B = nn.size(z, 0)
      def inner(i):
        z0 = tf.expand_dims(z[i], 0)
        y0 = tf.expand_dims(y[i], 0)
        out = generator(z0, y0)
        return out
      out = tf.map_fn(inner, tf.range(B),
          infer_shape=True, dtype=tf.float32, parallel_iterations=B)
      out = tf.squeeze(out, 1)
      return out
    elif True and batch_size % 8 == 0:
      B = nn.size(z, 0)
      z = tf.reshape(z, [8, -1] + nn.size(z)[1:])
      y = tf.reshape(y, [8, -1] + nn.size(y)[1:])
      #import pdb; pdb.set_trace()
      #z = tpu_feed.xla_sharding.tile(z, np.array([i % 8 for i in range(8)], dtype=np.int32))
      #y = tpu_feed.xla_sharding.tile(y, np.array([i % 8 for i in range(8)], dtype=np.int32))
      def inner(i):
        z0 = z[i]
        y0 = y[i]
        # z0 = tpu_feed.xla_sharding.assign_device(z0, i)
        # y0 = tpu_feed.xla_sharding.assign_device(y0, i)
        out = generator(z0, y0)
        return out
      dims = 8
      indices = tf.range(dims)
      tile_assignment = np.arange(np.prod(dims)).reshape(dims)
      # indices = tpu_feed.xla_sharding.tile(tensor=indices,
      #     tile_assignment=tile_assignment,
      #     assign_tuple_sharding=False)
      out = tf.map_fn(inner, indices,
          infer_shape=True, dtype=tf.float32, parallel_iterations=dims)
      #out = nn.view(out, -1, *nn.size(out)[2:])
      return out
      #import pdb; pdb.set_trace()
      #return tf.map_fn(generator, z, y)
    elif batch_size % 8 == 0:
      B = nn.size(z, 0)
      z = tf.reshape(z, [8, -1] + nn.size(z)[1:])
      y = tf.reshape(y, [8, -1] + nn.size(y)[1:])
      z = tpu_feed.xla_sharding.split(z, 0, 8, assign_tuple_sharding=False)
      y = tpu_feed.xla_sharding.split(y, 0, 8, assign_tuple_sharding=False)
    elif batch_size % 8 == 0:
      z = tpu_feed.xla_sharding.split(z, 0, 8, assign_tuple_sharding=False)
      y = tpu_feed.xla_sharding.split(y, 0, 8, assign_tuple_sharding=False)
      return generator(z, y)
    elif batch_size % 8 == 0:
      import pdb; pdb.set_trace()
      z = tf.reshape(z, [8, -1] + nn.size(z)[1:])
      y = tf.reshape(y, [8, -1] + nn.size(y)[1:])
      z = tpu_feed.xla_sharding.split(z, 0, 8, assign_tuple_sharding=True)
      y = tpu_feed.xla_sharding.split(y, 0, 8, assign_tuple_sharding=True)
      return tf.map_fn(generator, z, y)
    else:
      assert False
    return generator(z, y)

  def generate3(self, batch_size=4*64):
    self.model = biggan.BigGAN256( disc=False, use_ema=False );
    return self.generate2(generator=self.model.generator, batch_size=batch_size)

  def generate(self, *args):
    features = tf.nest.pack_sequence_as( self.input.structure, tf.nest.map_structure(tf.squeeze, args) )
    if False:
      samples = []
      for i in range(4):
        z = tf.expand_dims(features['z'][i], 0)
        y = tf.expand_dims(features['y'][i], 0)
        print(z, y)
        #import pdb; pdb.set_trace()
        sample = self.model.generator(z, y)
        samples.append(sample)
      #return [samples]
    else:
      z = features['z']
      y = features['y']
      samples = self.model.generator(z, y)
      return samples


  @property
  def z_dim(self):
    return self.model.generator.dim_z

  @property
  def num_classes(self):
    return self.model.generator.n_class

  def z_sample(self, batch_size=None, truncation=1.0, *, seed=None, numpy=False):
    if numpy:
      return truncated_z_sample(batch_size=batch_size, z_dim=self.z_dim, truncation=truncation, seed=seed)
    else:
      assert batch_size is not None
      z = tf.random.normal(shape=[batch_size, self.z_dim])
      if truncation != 1.0:
        z = z * truncation
      return z

  def y_sample(self, batch_size=None, *, seed=None, hot=True, numpy=False):
    if numpy:
      return sample_labels(batch_size=batch_size, num_classes=self.num_classes, seed=seed, hot=hot)
    else:
      assert batch_size is not None
      labels = tf.random.uniform(minval=0, maxval=self.num_classes, shape=[batch_size], dtype=tf.int32)
      if hot:
        return tf.one_hot(labels, self.num_classes, dtype=tf.float32)
      else:
        return labels

  def sample(self, batch_size=None, *, seed=None, numpy=False):
    return {
        'z': self.z_sample(batch_size=batch_size, seed=seed, numpy=numpy),
        'y': self.y_sample(batch_size=batch_size, seed=seed, numpy=numpy),
        }
  
  def enqueue(self):
    pass


