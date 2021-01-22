import os
import sys
sys.path += [os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')]

import tensorflow as tf
import tensorflow.compat.v1 as tf1

from absl import flags
from absl.testing import parameterized

from gaping import test_utils
from gaping import wrapper

from gaping.models import inception, inception_utils

import numpy as np
import tqdm

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

class InceptionTest(parameterized.TestCase, test_utils.GapingTestCase):

  def setUp(self):
    super().setUp()

  def test_001_tpu_session(self):
    with self.session().as_default() as session:
      devices = session.list_devices()
      for device in devices:
        self.log(device)
      self.assertTrue(any([':TPU:' in device.name for device in devices]))

  def test_002_inception(self):
    with self.session().as_default() as session:
      raw = inception.Inception3().eval()
      model = inception_utils.WrapInception(raw)
      saver = tf.train.Saver(var_list=tf.global_variables())
      saver.restore(session, 'gs://ml-euw4/models/inception_v3.ckpt')
      images_in = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='images_in')
      images_out = model(images_in)
      def get_fid(reals, fakes):
        pool1 = session.run(images_out[0], {images_in: reals})
        pool2 = session.run(images_out[0], {images_in: fakes})
        mean, cov = inception_utils.numpy_calculate_fid(pool1, pool2)
        print(mean, cov)
        return mean + cov
      def measure_fid(fid_low, fid_high, reals, fakes):
        fid = get_fid(reals, fakes)
        fid_low = either(fid_low, -float('inf'))
        fid_high = either(fid_high, float('inf'))
        self.assertAllInRange([fid], fid_low, fid_high)
        return fid
      measure_fid(-1, 1, ones(4,1,1,3), ones(4,1,1,3))
      # 208.2907296487286
      measure_fid(208, 209, ones(4,1,1,3), -ones(4,1,1,3))
      measure_fid(50, 100, randn(32,1,1,3), ones(32,1,1,3))
      measure_fid(-1, 1, ones(32,1,1,3), ones(32,1,1,3))
      for S in tqdm.tqdm([1,2,4,8,16,32]):
        measure_fid(0, 10, randn(32,S,S,3), randn(32,S,S,3))


if __name__ == "__main__":
  with wrapper.patch_tensorflow():
    tf.test.main()


