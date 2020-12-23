import tensorflow as tf

from gaping.tpu_topology_test import *
from gaping.biggan_test import *

if __name__ == "__main__":
  import gaping.wrapper
  with gaping.wrapper.patch_tensorflow():
    tf.test.main()

