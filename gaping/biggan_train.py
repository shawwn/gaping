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


def main(opts):
  globals()['opts'] = opts

  util.run_code(opts.code, 'gin_script.py', globals(), globals())
  

  assert False
  assert False

if __name__ == '__main__':
  parser = util.prepare_parser()
  util.run_app(parser, main)

