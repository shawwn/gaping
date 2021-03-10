import tensorflow as tf

from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import parsing_ops


def save_iterator(iterator_resource):
  if hasattr(iterator_resource, '_iterator_resource'):
    iterator_resource = iterator_resource._iterator_resource
  iterator_state_variant = gen_dataset_ops.serialize_iterator(
      iterator_resource)
  return parsing_ops.serialize_tensor(iterator_state_variant)


def restore_iterator(iterator_resource, serialized):
  if hasattr(iterator_resource, '_iterator_resource'):
    iterator_resource = iterator_resource._iterator_resource
  iterator_state_variant = parsing_ops.parse_tensor(serialized, tf.variant)
  restore_op = gen_dataset_ops.deserialize_iterator(iterator_resource, iterator_state_variant)
  return restore_op

