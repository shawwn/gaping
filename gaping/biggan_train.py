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


from . import util
from . import driver as driver_lib
from . import wrapper
from .util import EasyDict
from . import tftorch as nn

from .optim import optimizer

import tqdm

import tensorflow as tf
import numpy as np

def loss_hinge_dis(dis_fake, dis_real):
  loss_real = nn.mean(nn.relu(1. - dis_real))
  loss_fake = nn.mean(nn.relu(1. - dis_fake))
  return loss_real, loss_fake

def loss_hinge_gen(dis_fake):
  loss = -nn.mean(dis_fake)
  return loss

generator_loss = loss_hinge_gen
discriminator_loss = loss_hinge_dis


def stop(*args):
  def fn(x):
    if callable(x):
      x = x()
    return tf.stop_gradient(x)
  if len(args) <= 1:
    return fn(args[0])
  else:
    return [fn(x) for x in args]

def for_(n, body, *args, back_prop=False, parallel_iterations=1, **kws):
  results = tf.while_loop(
      lambda *args: True,
      body,
      [tf.convert_to_tensor(0, tf.int32)] + list(args),
      maximum_iterations=n,
      back_prop=back_prop,
      parallel_iterations=parallel_iterations,
      **kws)
  return results

class G_D(nn.Module):
  def __init__(self, G, D, scope='', **kws):
    super().__init__(scope=scope, **kws)
    self.G = G
    self.D = D

  def forward(self, z, y, x=None, dy=None, train_G=False, return_G_z=False, split_D=True):
    # Get Generator output given noise
    G_z = self.G(z, y)
    # If not training G, disable gradients
    if not train_G:
      G_z = stop(G_z)
    # Split_D means to run D once with real data and once with fake,
    # rather than concatenating along the batch dimension.
    if split_D:
      D_fake = self.D(G_z, y)
      if x is not None:
        D_real = self.D(x, dy)
        return D_fake, D_real
      else:
        if return_G_z:
          return D_fake, G_z
        else:
          return D_fake
    else:
      raise NotImplementedError()


class TrainGAN(biggan.GAN):
  def __init__(self, options, features):
    super().__init__()
    self.GD = G_D(self.gan.generator, self.gan.discriminator)
    self.G_optim = optimizer.SGD(self.gan.generator.parameters(), lr=options.G_lr)
    self.D_optim = optimizer.SGD(self.gan.discriminator.parameters(), lr=options.D_lr)

    self.train_op = tf.tpu.batch_parallel(
        lambda image: self.model_step({'image': image}),
        inputs=[features.image],
        num_shards=1,
        device_assignment=driver_lib.get().device_assignment([0]),
        )[0]

  @property
  def num_classes(self):
    return self.gan.generator.n_class

  def sample_z(self, batch_size=1):
    shape = [self.gan.generator.dim_z]
    if batch_size is not None:
      shape = [batch_size] + shape
    return stop(nn.truncated_normal(shape, mean=0.0, std=1.0))

  def sample_y(self, batch_size=1, maxval=1):
    shape = []
    if batch_size is not None:
      shape = [batch_size] + shape
    if maxval is None:
      maxval = self.gan.generator.n_class
    labels = tf.random.uniform(shape, minval=0, maxval=maxval, dtype=tf.int32)
    return stop(tf.one_hot(labels, self.gan.generator.n_class))

  def model_fn(self, features, *args):
    features = EasyDict(features)

    # Zero G's gradients by default before training G, for safety
    with tf.control_dependencies(self.G_optim.zero_grad()):

      # If accumulating gradients, loop multiple times before an optimizer step

      
      # If accumulating gradients, loop multiple times
      if True:
        def train_G_body(accumulation_index):
          z = self.sample_z()
          y = self.sample_y()
          D_fake = self.gan.generator(z, y)
          G_loss = generator_loss(D_fake) / float(options.num_G_accumulations)
          with tf.control_dependencies(self.G_optim.backward(G_loss)):
            return accumulation_index + 1
        op = for_(options.num_G_accumulations, train_G_body)
        ops = [op]
      else:
        ops = []
        for accumulation_index in range(options.num_G_accumulations):
          with tf.control_dependencies(ops):
            z = self.sample_z()
            y = self.sample_y()
            D_fake = self.GD(z, y, train_G=True)
            G_loss = generator_loss(D_fake) / float(options.num_G_accumulations)
            ops.extend(self.G_optim.backward(G_loss))

    with tf.control_dependencies(ops):
      ops = self.G_optim.step()

    with tf.control_dependencies(ops):
      return tf.no_op()

  def model_step(self, features):
    return self.model_fn(features)

  def run(self, op):
    driver = driver_lib.get()
    return driver.session.run(op)

  def fit(self):
    print('Initializing...')
    #self.run(self.gan.initializer)

    gs = tf.train.get_or_create_global_step()

    i = gs.initialized_value().eval()
    print('Training from step {}...'.format(i))
    while True:
      result = self.run(self.train_op)
      print(
          'Step {}'.format(i),
          self.GD.G.ScaledCrossReplicaBN.bn.num_batches_tracked.eval(),
          self.GD.G.ScaledCrossReplicaBN.bn.running_mean.eval(),
          )
      i += 1
      gs.load(i)

def main(opts):
  globals()['options'] = opts

  util.run_code(opts.startup, 'options.startup.py', globals(), globals())
  util.run_code(opts.custom, 'options.custom.py', globals(), globals())

  features = nxt

  trainer = TrainGAN(options=options, features=features)

  trainer.fit()
  
  assert False
  assert False

if __name__ == '__main__':
  np.set_printoptions(suppress=True)
  parser = util.prepare_parser()
  util.run_app(parser, main)

