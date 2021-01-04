import tensorflow as tf

if __name__ == "__main__":
  import gaping.wrapper
  with gaping.wrapper.patch_tensorflow():
    tf.test.main()

