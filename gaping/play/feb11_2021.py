import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import gen_ragged_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import map_fn


def _restack_sparse_tensor_logically(indices, values, shape):
  sparse_tensor_rank = indices.get_shape().dims[-1].value
  if sparse_tensor_rank is not None:
    sparse_tensor_rank += 1

  def fn(args):
    res = gen_sparse_ops.serialize_sparse(
        args[0], args[1], args[2], out_type=dtypes.variant)
    return res

  # Applies a map function to the component tensors to serialize each
  # sparse tensor element and batch them all, then deserializes the batch.
  # TODO(rachelim): Try to do this without map_fn -- add the right offsets
  # to shape and indices tensors instead.
  result = map_fn.map_fn(
      fn, [indices, values, shape], dtype=dtypes.variant)
  return sparse_ops.deserialize_sparse(
      result, dtype=values.dtype, rank=sparse_tensor_rank)
  
def serialize_sparse(indices, values, shape):
  indices = tf.convert_to_tensor(indices, dtype=tf.int64)
  values = tf.convert_to_tensor(values)
  sparse_tensor_rank = indices.get_shape().dims[-1].value
  if sparse_tensor_rank is not None:
    sparse_tensor_rank += 1

  def fn(args):
    res = gen_sparse_ops.serialize_sparse(
        args[0], args[1], args[2], out_type=dtypes.variant)
    return res
  return fn([indices, values, shape])

def tf_io_encode_raw_i64(tokens):
  unit_size = tokens.dtype.size
  total_size = tf.size(tokens, out_type=tf.int64) * unit_size
  serialized = tf.serialize_tensor(tokens)
  serialized_size = tf.size(tf.strings.bytes_split(serialized), out_type=tf.int64)
  offset = serialized_size - total_size
  return tf.strings.substr(serialized, offset, -1)

def tf_io_encode_raw_string(string):
  unit_size = tokens.dtype.size
  total_size = tf.size(tokens, out_type=tf.int64) * unit_size
  serialized = tf.serialize_tensor(tokens)
  serialized_size = tf.size(tf.strings.bytes_split(serialized), out_type=tf.int64)
  offset = serialized_size - total_size
  return tf.strings.substr(serialized, offset, -1)


# >>> table = tf.lookup.experimental.DenseHashTable(key_dtype=tf.int64, value_dtype=tf.variant, default_value=feb11_2021.serialize_sparse( [[0, 0], [1, 2]], [1, 2], [3, 4] ), empty_key=-1, deleted_key=-2, checkpoint=False)
# >>> table.size()
# <tf.Tensor 'MutableDenseHashTable_2_Size/LookupTableSizeV2:0' shape=() dtype=int64>
# >>> r( table.size() )
# 0
# >>> table.lookup( -21 )
# <tf.Tensor 'MutableDenseHashTable_2_lookup_table_find/LookupTableFindV2:0' shape=(3,) dtype=variant>
# >>> table.insert( -21, feb11_2021.serialize_sparse( [[0]], [21], [1] ) )
# <tf.Operation 'MutableDenseHashTable_2_lookup_table_insert/LookupTableInsertV2' type=LookupTableInsertV2>
# >>> r( table.insert( -21, feb11_2021.serialize_sparse( [[0]], [21], [1] ) ) )
# >>> table.lookup( -21 )
# <tf.Tensor 'MutableDenseHashTable_2_lookup_table_find_1/LookupTableFindV2:0' shape=(3,) dtype=variant>
# >>> sparse_ops.deserialize_sparse( table.lookup( -21 ), tf.int32 ).values
# <tf.Tensor 'DeserializeSparse_8:1' shape=(?,) dtype=int32>
# >>> r( sparse_ops.deserialize_sparse( table.lookup( -21 ), tf.int32 ).values )
# array([21], dtype=int32)
# >>> r( sparse_ops.deserialize_sparse( table.lookup( -20 ), tf.int32 ).values )
# array([1, 2], dtype=int32)
# >>> tf.size( sparse_ops.deserialize_sparse( table.lookup( -20 ), tf.int32 ).values )
# <tf.Tensor 'Size:0' shape=() dtype=int32>
# >>> r( tf.size( sparse_ops.deserialize_sparse( table.lookup( -20 ), tf.int32 ).values ) )
# 2

# >>> r( tf.strings.reduce_join( tf.sparse.to_dense( sparse_ops.deserialize_sparse( feb11_2021.serialize_sparse( [[0], [1]], [b'foo', b'bar'], [tf.convert_to_tensor(2)] ), tf.string ) ) ) )
# b'foobar'

