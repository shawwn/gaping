from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.core.framework import attr_value_pb2

import google.protobuf.json_format as jf

def parse_xla_sharding(op, as_dict=True):
  if hasattr(op, 'op'):
    op = op.op
  proto = xla_data_pb2.OpSharding();
  proto.ParseFromString( op.get_attr('_XlaSharding') );
  if as_dict:
    return jf.MessageToDict(proto)
  return proto
