import os
import sys
sys.path += [os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')]

import tensorflow as tf
import tensorflow.compat.v1 as tf1

from absl import flags
from absl.testing import parameterized

from gaping import test_utils
from gaping import wrapper

from gaping.util import EasyDict
from gaping.models import inception, inception_utils

import numpy as np
import random
import tqdm

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch
import gaping.tftorch as nn

from scipy.stats import describe

def shuffled(x):
  if isinstance(x, (tuple, list)):
    x = np.stack(x)
  x = x.copy()
  random.shuffle(x)
  return x

def zeros(*shape):
  return np.zeros(shape)

def ones(*shape):
  return np.ones(shape)

def randn(*shape, minval=-1.0, maxval=1.0):
  return np.random.uniform(size=shape, low=minval, high=maxval)

def either(a, b):
  if a is None:
    return b
  return a


from PIL import Image

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


from argparse import ArgumentParser

def prepare_parser():
  usage = 'Parser for all scripts.'
  parser = ArgumentParser(description=usage)
  return parser
  
def add_fid_parser(parser):
  ### Dataset/Dataloader stuff ###
  parser.add_argument(
    'reals', type=str,
    help='The folder containing real images')
  parser.add_argument(
    'fakes', type=str, default=None, nargs='?',
    help='The folder containing fake images')
  parser.add_argument(
    '--extensions', type=str, default='.jpg,.jpeg,.png,.ppm,.bmp,.pgm,.tif,.tiff,.webp',
    help='Supported extensions for image files'
         '(default: %(default)s)')
  parser.add_argument(
    '--num_workers', type=int, default=0,
    help='Number of image loader workers'
         '(default: %(default)s)')
  parser.add_argument(
    '--batch_size', type=int, default=16,
    help='FID batch size'
         '(default: %(default)s)')
  parser.add_argument(
    '--save_real_activations', type=str, default='activations.npy',
    help='Save real activations filepath (relative to reals)'
         '(default: %(default)s)')
  return parser


class CenterCropLongEdge(object):
  """Crops the given PIL Image on the long edge.
  Args:
      size (sequence or int): Desired output size of the crop. If size is an
          int instead of sequence like (h, w), a square crop (size, size) is
          made.
  """
  def __call__(self, img):
    """
    Args:
        img (PIL Image): Image to be cropped.
    Returns:
        PIL Image: Cropped image.
    """
    return transforms.functional.center_crop(img, min(img.size))

  def __repr__(self):
    return self.__class__.__name__



class RandomCropLongEdge(object):
  """Crops the given PIL Image on the long edge with a random start point.
  Args:
      size (sequence or int): Desired output size of the crop. If size is an
          int instead of sequence like (h, w), a square crop (size, size) is
          made.
  """
  def __call__(self, img):
    """
    Args:
        img (PIL Image): Image to be cropped.
    Returns:
        PIL Image: Cropped image.
    """
    size = (min(img.size), min(img.size))
    # Only step forward along this edge if it's the long edge
    i = (0 if size[0] == img.size[0]
          else np.random.randint(low=0,high=img.size[0] - size[0]))
    j = (0 if size[1] == img.size[1]
          else np.random.randint(low=0,high=img.size[1] - size[1]))
    return transforms.functional.crop(img, i, j, size[0], size[1])

  def __repr__(self):
    return self.__class__.__name__



class ToPILImage(object):
  """Converts the given tensor to a PIL image
  Args:
      size (sequence or int): Desired output size of the crop. If size is an
          int instead of sequence like (h, w), a square crop (size, size) is
          made.
  """
  def __call__(self, input):
    """
    Args:
        input: tensor to be converted.
    Returns:
        PIL Image: Converted image.
    """
    input = input.numpy()
    input *= 0.5
    input += 0.5
    input *= 256
    input = np.clip(input, 0.0, 255.0)
    input = input.astype(np.uint8)
    input = np.transpose(input, [1,2,0])
    input = Image.fromarray(input)
    return input
  def __repr__(self):
    return self.__class__.__name__


# fall back to while loops for pfor
from tensorflow.python.ops.parallel_for.pfor import flags as pfor_flags
pfor_flags.FLAGS.op_conversion_fallback_to_while_loop = True

from tensorflow.python.ops.parallel_for import control_flow_ops as pfor_control_flow_ops

def loop_n(thunk, n, **kws):
  return pfor_control_flow_ops.pfor(thunk, iters=n, **kws)


class InputQueue:
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

from io import BytesIO

def pil_to_bytes(pil, format='JPEG', quality=95):
  with BytesIO() as bio:
    if format in ['JPG', 'JPEG']:
      pil.save(bio, format=format, quality=quality)
    else:
      pil.save(bio, format=format)
    return bio.getvalue()

class ImageQueue:
  def __init__(self, shuffle=False, batch_size=1, transform=None, **kws):
    self.input = InputQueue(dtypes=[tf.string], shuffle=shuffle, **kws)
    self.batch_size = batch_size
    self.transform = transform
    self.image_in = tf.placeholder(tf.string, shape=[None], name="image_in")
    self.enqueue_op = self.input.enqueue_many(self.image_in, name="enqueue_image")
    self.dequeue_op = self.input.dequeue(n=batch_size, transform=self.bytes_to_image, name="dequeue_image")
    self.size_op = self.input.queue.size()
  def bytes_to_image(self, image_bytes):
    image = tf.io.decode_image(image_bytes)
    image = tf.cast(image, tf.float32)/255.0 * 2.0 - 1.0
    if self.transform:
      image = self.transform(image)
    return image
  @property
  def size(self):
    return self.size_op.eval()
  def dequeue(self):
    return self.dequeue_op.eval()
  def enqueue(self, images, session=None):
    session = session or tf1.get_default_session()
    if not isinstance(images, (tuple, list)):
      images = [images]
    input = [pil_to_bytes(img) for img in images]
    session.run(self.enqueue_op, {self.image_in: input})
    return len(input), sum([len(x) for x in input])

def sequential(n, op):
  ops = []
  for i in range(n):
    with tf.control_dependencies(ops):
      ops += [op()]
  return ops

class MeasureFID(parameterized.TestCase, test_utils.GapingTestCase):

  def setUp(self):
    super().setUp()
    # parse command line and run    
    parser = prepare_parser()
    parser = add_fid_parser(parser)
    config = vars(parser.parse_args())
    config = EasyDict(config)
    self.args = config


    self.transform = transforms.Compose([
      transforms.Resize((299, 299), Image.ANTIALIAS),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
      #transforms.Normalize((0.0, 0.0, 0.0), (2.0, 2.0, 2.0)),
      #transforms.ToPILImage(),
      ])

    extensions = self.args.extensions.split(',')
    extensions += self.args.extensions.upper().split(',')
    extensions = tuple(extensions)
    self.reals_set = datasets.DatasetFolder(self.args.reals, transform=self.transform, target_transform=None, extensions=extensions, loader=pil_loader)
    self.fakes_set = datasets.DatasetFolder(self.args.fakes, transform=self.transform, target_transform=None, extensions=extensions, loader=pil_loader) if self.args.fakes else None
    self.reals_loader = DataLoader(dataset=self.reals_set, num_workers=self.args.num_workers, batch_size=self.args.batch_size, shuffle=True)
    self.fakes_loader = DataLoader(dataset=self.fakes_set, num_workers=self.args.num_workers, batch_size=self.args.batch_size, shuffle=True) if self.args.fakes else None
    self.sess = self.cached_session(interactive=True)
    self.model = inception_utils.WrapInception(inception.Inception3().eval(), resize_mode=None)
    self.saver = tf.train.Saver(var_list=tf.global_variables())
    self.saver.restore(self.sess, 'gs://ml-euw4/models/inception_v3.ckpt')
    self.batch_size = 8
    self.reals_infeed = ImageQueue(batch_size=self.batch_size, transform=lambda image: self.model(image)[0], shuffle=False)
    self.fakes_infeed = ImageQueue(batch_size=self.batch_size, transform=lambda image: self.model(image)[0], shuffle=False) if self.args.fakes else None

        

  # def test_001_tpu_session(self):
  #   with self.session().as_default() as session:
  #     devices = session.list_devices()
  #     for device in devices:
  #       self.log(device)
  #     self.assertTrue(any([':TPU:' in device.name for device in devices]))

  def main(self):
    with self.cached_session().as_default() as self.sess:
      #       # saver = tf.train.Saver(var_list=tf.global_variables())
      # saver.restore(self.sess, 'gs://ml-euw4/models/inception_v3.ckpt')
      # self.images_in_nchw = tf.placeholder(tf.float32, shape=[None, 3, None, None], name='images_in')
      # self.images_out = self.model(tf.transpose(self.images_in_nchw, [0,2,3,1]))
      #import pdb; pdb.set_trace()
      print(self.args)
      real_count = 0
      fake_count = 0
      total_bytes = 0
      self.fid = float('inf'), float('inf')
      self.reals_it = enumerate(self.reals_loader)
      self.fakes_it = enumerate(self.fakes_loader) if self.args.fakes else None
      self.reals_pool = []
      self.fakes_pool = [] if self.args.fakes else None
      with tqdm.tqdm(total=min(len(self.reals_set.samples), len(self.fakes_set.samples) if self.args.fakes else len(self.reals_set.samples)), position=0) as pbar, \
           tqdm.tqdm(total=1, position=1) as ibar:
        while True:
          try:
            real_batch_idx, (real_features, real_labels) = next(self.reals_it)
            if self.args.fakes:
              fake_batch_idx, (fake_features, fake_labels) = next(self.fakes_it)
            try:
              # save_image(torch.cat([real_features, fake_features])*0.5+0.5, 'test_tmp.jpg')
              # os.rename('test_tmp.jpg', 'test.jpg')
              real_n, real_bytes = self.reals_infeed.enqueue([ToPILImage()(x) for x in real_features])
              if self.args.fakes:
                fake_n, fake_bytes = self.fakes_infeed.enqueue([ToPILImage()(x) for x in fake_features])
              real_count += real_n
              if self.args.fakes:
                fake_count += fake_n
              total_bytes += real_bytes
              if self.args.fakes:
                total_bytes += fake_bytes
              while real_count >= self.batch_size and (fake_count >= self.batch_size or not self.args.fakes):
                ibar.update(self.batch_size)
                if self.args.fakes:
                  real_pool, fake_pool = self.sess.run([self.reals_infeed.dequeue_op, self.fakes_infeed.dequeue_op])
                  for real, fake in zip(real_pool, fake_pool):
                    self.reals_pool += [real[0]]
                    self.fakes_pool += [fake[0]]
                  real_count -= self.batch_size
                  fake_count -= self.batch_size
                  self.fid = inception_utils.numpy_calculate_fid(self.reals_pool[-128:], self.fakes_pool[-128:])
                else:
                  real_pool = self.sess.run(self.reals_infeed.dequeue_op)
                  for real in real_pool:
                    self.reals_pool += [real[0]]
                  real_count -= self.batch_size
              if real_batch_idx % 8 == 0 and self.args.fakes:
                self.fid2 = inception_utils.numpy_calculate_fid(self.reals_pool, self.fakes_pool, rowvar=False)
                ibar.set_description('Traditional FID: %s + %s' % (nn.num(self.fid2[0], 3), nn.num(self.fid2[1], 3)))
              #pbar.update(self.args.batch_size)
              pbar.update(real_n)
              pbar.set_description('Short FID: %s + %s | Uploaded %sMB' % (nn.num(self.fid[0], 3), nn.num(self.fid[1], 3), nn.num(total_bytes/1024/1024, 3)))
              if self.args.save_real_activations:
                np.save(os.path.join(self.args.reals, self.args.save_real_activations), np.stack(self.reals_pool))
            except KeyboardInterrupt:
              import pdb; pdb.set_trace()
              print(self.args)
          except KeyboardInterrupt:
            import pdb; pdb.set_trace()
            print(self.args)
          except StopIteration:
            if self.args.save_real_activations:
              np.save(os.path.join(self.args.reals, self.args.save_real_activations), np.stack(self.reals_pool))
            self.fid = inception_utils.numpy_calculate_fid(self.reals_pool, self.fakes_pool)
            print('Short FID', self.fid)
            self.fid2 = inception_utils.numpy_calculate_fid(self.reals_pool, self.fakes_pool, rowvar=False)
            print('Traditional FID', self.fid2)
            import pdb; pdb.set_trace()
            k = 128; self.fid = inception_utils.numpy_calculate_fid(random.choices(self.reals_pool, k=k), random.choices(self.fakes_pool, k=k)); print(self.fid)


if __name__ == "__main__":
  with wrapper.patch_tensorflow():
    measure_fid = MeasureFID()
    measure_fid.setUp()
    measure_fid.main()

