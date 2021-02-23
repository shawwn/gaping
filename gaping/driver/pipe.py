from .. import driver as driver_lib

import tensorflow as tf

from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops

from . import padded_long
from . import sequence
from . import util


class PipeBuilder:
  def __init__(self, stager=None):
    self.dtypes = []
    self.shapes = []
    self.names = []
    if stager is not None:
      for name, dtype, shape in zip(stager.names, stager.dtypes, stager.shapes):
        self.input(name=name, dtype=dtype, shape=shape)

  def input(self, name, dtype, shape=()):
    assert name not in self.names
    self.dtypes.append(dtype)
    self.names.append(name)
    self.shapes.append(shape)
    return len(self.dtypes) - 1

  def get(self, ordered=True, **kws):
    assert len(self.dtypes) > 0
    shared_name = kws.pop('shared_name', driver_lib.get().uniq('pipe'))
    with ops.init_scope():
      stager = data_flow_ops.MapStagingArea(list(self.dtypes), shapes=list(self.shapes), names=list(self.names), shared_name=shared_name, ordered=ordered, **kws)
    self.dtypes.clear()
    self.shapes.clear()
    self.names.clear()
    return stager

