import sys
import os
sys.path += [os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))]

import gaping.driver
import gaping.wrapper

import tensorflow as tf

from gaping import tf_tools as tft

import numpy as np
import tqdm

from pprint import pprint as pp
from importlib import reload

def reset_session(session=None, graph=None, interactive=True, **kws):
  if session is None:
    session = tf.compat.v1.get_default_session()
  if graph is None:
    graph = tf.Graph()
  graph.as_default().__enter__()
  session2 = gaping.wrapper.clone_session(session, graph=graph, interactive=interactive, **kws)
  session2.as_default().__enter__()
  if 'driver' in globals():
    driver = globals()['driver']
    driver.session = session2
    session2.driver = driver
    globals()['r'] = driver.run
    globals()['shard'] = driver.shard
  return session2

if __name__ == '__main__':
  _tf_patch = gaping.wrapper.patch_tensorflow_interactive()
  driver = gaping.driver.new(interactive=True)
  r = driver.run
  shard = driver.shard
  
  
