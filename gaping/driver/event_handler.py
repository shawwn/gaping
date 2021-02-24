from .. import driver as driver_lib
from .. import tf_tools as tft

import tensorflow as tf

# from . import padded_long
# from . import sequence
# from . import util
# from . import pipe
# from . import latch as latch_lib


class EventHandler:
  def on_event(self, event, sequence, end_of_batch):
    raise NotImplementedError()


class LatchEventHandler(EventHandler):
  def __init__(self, latch):
    self.latch = latch

  def on_event(self, event, sequence, end_of_batch):
    return self.latch.count_down()
  


