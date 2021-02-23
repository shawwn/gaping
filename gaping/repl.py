import sys
import os
sys.path += [os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))]

import gaping.driver
import gaping.wrapper

import tensorflow as tf

from pprint import pprint as pp
from importlib import reload

if __name__ == '__main__':
  gaping.wrapper.patch_tensorflow_interactive()
  driver = gaping.driver.new(interactive=True)
  res = getattr(driver, 'resolver', None)
  graph = driver.graph
  sess = driver.session
  r = driver.run
  
  
