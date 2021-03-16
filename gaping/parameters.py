import gin
import gin.tf.external_configurables

import tensorflow as tf

from .util import EasyDict

@gin.configurable
def options(**kws):
  return EasyDict(kws)
