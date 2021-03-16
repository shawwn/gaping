from . import biggan
from . import tftorch as nn

import numpy as np
import os

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
  loss_fake = nn.mean(nn.relu(1. + dis_fake))
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
        if return_G_z:
          return D_fake, D_real, G_z
        else:
          return D_fake, D_real
      else:
        if return_G_z:
          return D_fake, G_z
        else:
          return D_fake
    else:
      raise NotImplementedError()

def image_grid(input_tensor, grid_shape, image_shape=(32, 32), num_channels=3):
  """Arrange a minibatch of images into a grid to form a single image.

  Args:
    input_tensor: Tensor. Minibatch of images to format, either 4D
        ([batch size, height, width, num_channels]) or flattened
        ([batch size, height * width * num_channels]).
    grid_shape: Sequence of int. The shape of the image grid,
        formatted as [grid_height, grid_width].
    image_shape: Sequence of int. The shape of a single image,
        formatted as [image_height, image_width].
    num_channels: int. The number of channels in an image.

  Returns:
    Tensor representing a single image in which the input images have been
    arranged into a grid.

  Raises:
    ValueError: The grid shape and minibatch size don't match, or the image
        shape and number of channels are incompatible with the input tensor.
  """
  if grid_shape[0] * grid_shape[1] != int(input_tensor.shape[0]):
    raise ValueError("Grid shape %s incompatible with minibatch size %i." %
                     (grid_shape, int(input_tensor.shape[0])))
  if len(input_tensor.shape) == 2:
    num_features = image_shape[0] * image_shape[1] * num_channels
    if int(input_tensor.shape[1]) != num_features:
      raise ValueError("Image shape and number of channels incompatible with "
                       "input tensor.")
  elif len(input_tensor.shape) == 4:
    if (int(input_tensor.shape[1]) != image_shape[0] or
        int(input_tensor.shape[2]) != image_shape[1] or
        int(input_tensor.shape[3]) != num_channels):
      raise ValueError("Image shape and number of channels incompatible with "
                       "input tensor. %s vs %s" % (
                           input_tensor.shape, (image_shape[0], image_shape[1],
                                                num_channels)))
  else:
    raise ValueError("Unrecognized input tensor format.")
  height, width = grid_shape[0] * image_shape[0], grid_shape[1] * image_shape[1]
  input_tensor = tf.reshape(input_tensor, tuple(grid_shape) + tuple(image_shape) + (num_channels,))
  input_tensor = tf.transpose(a=input_tensor, perm=[0, 1, 3, 2, 4])
  input_tensor = tf.reshape(input_tensor, [grid_shape[0], width, image_shape[0], num_channels])
  input_tensor = tf.transpose(a=input_tensor, perm=[0, 2, 1, 3])
  input_tensor = tf.reshape(input_tensor, [1, height, width, num_channels])
  return input_tensor



class TrainGAN(biggan.GAN):
  def __init__(self, options, dataset):
    super().__init__()
    self.iterator = dataset.make_initializable_iterator()
    self.features = EasyDict(self.iterator.get_next())
    self.global_step = tf.train.get_or_create_global_step()
    self.options = options
    self.GD = G_D(self.gan.generator, self.gan.discriminator)
    self.G_lr = nn.localvar('G_lr', shape=[], dtype=tf.float32)
    self.D_lr = nn.localvar('D_lr', shape=[], dtype=tf.float32)
    self.fixed_z = nn.localvar('fixed_z', shape=[options.batch_size, self.gan.generator.dim_z], dtype=tf.float32)
    self.fixed_y = nn.localvar('fixed_y', shape=[options.batch_size, self.num_classes], dtype=tf.float32)
    self.G_optim = optimizer.SGD(self.gan.generator.parameters(), lr=self.G_lr)
    self.D_optim = optimizer.SGD(self.gan.discriminator.parameters(), lr=self.D_lr)
    self.G_optim_ops = [nn.mean(x) for x in self.G_optim.grad_list]
    self.D_optim_ops = [nn.mean(x) for x in self.D_optim.grad_list]
    self.train_op = self.compile(self.features)


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

  def model_fn(self, step_index, features, *args):
    features = EasyDict(features)

    self.GD.D.train()
    #self.GD.G.eval()
    self.GD.G.train()

    D_losses = tf.TensorArray(tf.float32, options.num_D_accumulations, element_shape=[])
    D_losses_real = tf.TensorArray(tf.float32, options.num_D_accumulations, element_shape=[])
    D_losses_fake = tf.TensorArray(tf.float32, options.num_D_accumulations, element_shape=[])
    G_losses = tf.TensorArray(tf.float32, options.num_G_accumulations, element_shape=[])

    D_reals = tf.TensorArray(tf.float32, options.num_D_accumulations, element_shape=[3, 256, 256])
    D_fakes = tf.TensorArray(tf.float32, options.num_D_accumulations, element_shape=[3, 256, 256])
    D_reals_prob = tf.TensorArray(tf.float32, options.num_D_accumulations, element_shape=[])
    D_fakes_prob = tf.TensorArray(tf.float32, options.num_D_accumulations, element_shape=[])

    with tf.control_dependencies(self.D_optim.zero_grad() + self.G_optim.zero_grad()):

      # If accumulating gradients, loop multiple times before an optimizer step
      def train_D_body(accumulation_index, D_losses, D_losses_real, D_losses_fake, D_reals, D_fakes, D_reals_prob, D_fakes_prob):
        i = accumulation_index
        z = self.sample_z()
        y = self.sample_y()
        x = stop(tf.expand_dims(features.image[options.num_D_accumulations*step_index + i], 0))
        D_fake, D_real, G_z = self.GD(z, y, x, y, train_G=False, return_G_z=True)

        # Compute components of D's loss, average them, and divide by 
        # the number of gradient accumulations
        D_loss_real, D_loss_fake = discriminator_loss(D_fake, D_real)
        # D_loss_real = tf.cond(D_loss_real < 0.1, lambda: 0.0, lambda: D_loss_real)
        # D_loss_fake = tf.cond(D_loss_fake < 0.1, lambda: 0.0, lambda: D_loss_fake)
        D_loss = (D_loss_real + D_loss_fake)
        D_loss = tf.cond(D_loss < 0.2, lambda: 0.0, lambda: D_loss)
        def check():
          return tf.convert_to_tensor(True)
          # return tf.logical_and(
          #     D_loss_real > 0.1,
          #     D_loss_fake > 0.1)
        def yes():
          return tf.group(self.D_optim.backward(D_loss / float(options['num_D_accumulations'])))
        def no():
          return tf.no_op()
        with tf.control_dependencies([tf.cond(check(), yes, no)]):
          return [accumulation_index + 1,
              D_losses.write(accumulation_index, D_loss),
              D_losses_real.write(accumulation_index, D_loss_real),
              D_losses_fake.write(accumulation_index, D_loss_fake),
              D_reals.write(accumulation_index, x[0]),
              D_fakes.write(accumulation_index, G_z[0]),
              D_reals_prob.write(accumulation_index, D_real[0]),
              D_fakes_prob.write(accumulation_index, D_fake[0]),
              ]
      _, D_losses, D_losses_real, D_losses_fake, D_reals, D_fakes, D_reals_prob, D_fakes_prob = for_(options.num_D_accumulations, train_D_body,
          D_losses, D_losses_real, D_losses_fake, D_reals, D_fakes, D_reals_prob, D_fakes_prob)
      D_losses = D_losses.stack()
      D_losses_real = D_losses_real.stack()
      D_losses_fake = D_losses_fake.stack()
      D_reals = D_reals.stack()
      D_fakes = D_fakes.stack()
      D_reals_prob = D_reals_prob.stack()
      D_fakes_prob = D_fakes_prob.stack()
      ops = [_, D_losses, D_losses_real, D_losses_fake, D_reals, D_fakes, D_reals_prob, D_fakes_prob]

    with tf.control_dependencies(ops):
      ops = self.D_optim.step()

    #self.GD.D.eval()
    self.GD.D.train()
    self.GD.G.train()

    # Zero G's gradients by default before training G, for safety
    with tf.control_dependencies(ops): # + self.G_optim.zero_grad()):
      
      # If accumulating gradients, loop multiple times
      def train_G_body(accumulation_index, G_losses):
        z = self.sample_z()
        y = self.sample_y()
        D_fake = self.GD(z, y, train_G=True)
        G_loss = generator_loss(D_fake) / float(options.num_G_accumulations)
        with tf.control_dependencies(self.G_optim.backward(G_loss)):
          return [accumulation_index + 1, G_losses.write(accumulation_index, G_loss)]
      _, G_losses = for_(options.num_G_accumulations, train_G_body, G_losses)
      G_losses = G_losses.stack()
      ops = [_, G_losses]

    with tf.control_dependencies(ops):
      ops = self.G_optim.step()

    with tf.control_dependencies(ops):
      self.GD.G.eval()
      self.GD.D.eval()
      fixed = stop(self.GD.G(self.fixed_z[0:8], self.fixed_y[0:8]))

    D_losses.set_shape([options.num_D_accumulations])
    D_losses_real.set_shape([options.num_D_accumulations])
    D_losses_fake.set_shape([options.num_D_accumulations])
    G_losses.set_shape([options.num_G_accumulations])
    D_reals.set_shape([options.num_D_accumulations, 3, 256, 256])
    D_fakes.set_shape([options.num_D_accumulations, 3, 256, 256])
    D_reals_prob.set_shape([options.num_D_accumulations])
    D_fakes_prob.set_shape([options.num_D_accumulations])
    return {
        'fixed': fixed,
        'D_losses': D_losses,
        'D_losses_real': D_losses_real,
        'D_losses_fake': D_losses_fake,
        'G_losses': G_losses,
        'D_reals': D_reals,
        'D_fakes': D_fakes,
        'D_reals_prob': D_reals_prob,
        'D_fakes_prob': D_fakes_prob,
        }

  def model_step(self, features):
    self.D_steps = self.options.batch_size // self.options.num_D_accumulations
    self.output_structure = self.model_fn(0, features)
    def model_body(i):
      with tf.control_dependencies(tf.nest.flatten(self.model_fn(i+1, features))):
        return i + 1
    # ops = []
    # if self.D_steps > 1:
    #   ops = [for_(self.D_steps - 1, model_body)]
    # with tf.control_dependencies(ops):
    #   self.output_structure = tf.nest.map_structure(tf.identity, self.output_structure)
    return tf.nest.flatten(self.output_structure)

  def to_grid(self, images):
    images = nn.permute(images, 0, 2, 3, 1)
    images = tf.image.resize(images, [64, 64],
        method=tf.image.ResizeMethod.AREA)
    images = image_grid(images, (2,4), (64, 64))[0]
    images = tf.io.encode_jpeg(tf.saturate_cast(255.0*(images*0.5+0.5), tf.uint8), quality=90)
    return images

  def compile(self, features):
    results = tf.tpu.batch_parallel( lambda image: self.model_step({'image': image}), inputs=[features.image], num_shards=1, device_assignment=driver_lib.get().device_assignment([0]),)
    outputs = tf.nest.pack_sequence_as(self.output_structure, results)
    for key in ['D_reals', 'D_fakes', 'fixed']:
      outputs[key] = self.to_grid(outputs[key])
    return outputs

  def run(self, op):
    driver = driver_lib.get()
    return driver.session.run(op)

  def fit(self):
    # print('Initializing TPU...')
    # self.run(tf.tpu.initialize_system())
    print('Initializing dataset...')
    self.run(self.iterator.initializer)
    print('Initializing GAN...')
    self.run(self.gan.generator.initializer)
    self.run(self.gan.discriminator.initializer)
    print('Loading global step...')
    self.global_step.load(0)
    print('Setting learning rates...')
    self.D_lr.load(self.options.D_lr)
    self.G_lr.load(self.options.G_lr)
    print('Loading fixed latents...')
    self.fixed_z.load(truncated_z_sample(options.batch_size, self.gan.generator.dim_z, seed=0))
    self.fixed_y.load(sample_labels(options.batch_size, self.num_classes, seed=0))

    print('Fetching step number...')
    i = self.global_step.eval()

    print('Training from step {}...'.format(i))
    while True:
      out = EasyDict(self.run(self.train_op))
      print('Step {}'.format(i))
      with open('reals.jpg', 'wb') as f: f.write(out.D_reals)
      with open('fakes.jpg', 'wb') as f: f.write(out.D_fakes)
      with open('fixed.jpg', 'wb') as f: f.write(out.fixed)
      print('G_losses', out.G_losses)
      print('D_losses', out.D_losses)
      print('D_losses_real', out.D_losses_real)
      print('D_losses_fake', out.D_losses_fake)
      print('D_reals_prob', out.D_reals_prob)
      print('D_fakes_prob', out.D_fakes_prob)
      print(self.GD.G.ScaledCrossReplicaBN.bn.num_batches_tracked.eval())
      # print(self.GD.G.ScaledCrossReplicaBN.bn.running_mean.eval())
      if os.path.exists('DEBUG'):
        import pdb; pdb.set_trace()
      if os.path.exists('REINIT_G'):
        os.unlink('REINIT_G')
        print('REINIT_G')
        self.run(self.gan.generator.initializer)
      if os.path.exists('REINIT_D'):
        os.unlink('REINIT_D')
        print('REINIT_D')
        self.run(self.gan.discriminator.initializer)
      if os.path.exists('DOUBLE_LR'):
        os.unlink('DOUBLE_LR')
        self.G_lr.load(self.G_lr.eval()*2);
        self.D_lr.load(self.D_lr.eval()*2);
        print('G_lr now', self.G_lr.eval())
        print('D_lr now', self.D_lr.eval())
      if os.path.exists('HALF_LR'):
        os.unlink('HALF_LR')
        self.G_lr.load(self.G_lr.eval()/2);
        self.D_lr.load(self.D_lr.eval()/2);
        print('G_lr now', self.G_lr.eval())
        print('D_lr now', self.D_lr.eval())
      i += 1
      self.global_step.load(i)

def main(opts):
  globals()['options'] = opts

  util.run_code(opts.startup, 'options.startup.py', globals(), globals())
  util.run_code(opts.custom, 'options.custom.py', globals(), globals())

  trainer = TrainGAN(options=options, dataset=dataset)

  trainer.fit()
  
  assert False
  assert False

if __name__ == '__main__':
  np.set_printoptions(suppress=True)
  parser = util.prepare_parser()
  util.run_app(parser, main)

