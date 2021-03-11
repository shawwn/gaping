import tensorflow as tf

from tensorflow.core.profiler import profiler_service_pb2
from tensorflow.core.profiler import profile_pb2

import google.protobuf.json_format as jf

from ..util import EasyDict

from .. import tf_tools as tft

from base64 import b64decode
import os
import threading
import time

kMaxEvents = 1000000

def build_profile_request(duration_ms=10000, repository_root=None, session_id=None, *, include_dataset_ops=False, max_events=kMaxEvents):
  opts = profiler_service_pb2.ProfileOptions(include_dataset_ops=include_dataset_ops)
  request = profiler_service_pb2.ProfileRequest()
  request.tools.append("op_profile");
  request.tools.append("input_pipeline");
  request.tools.append("memory_viewer");
  request.tools.append("overview_page");
  request.tools.append("pod_viewer");
  request.tools.append("trace_viewer");
  if session_id is not None:
    assert len(session_id) == 16
    request.session_id = session_id
  if repository_root and repository_root.startswith('gs://'):
    assert repository_root.startswith('gs://')
    request.repository_root = repository_root
  request.duration_ms = duration_ms
  request.max_events = max_events
  return request

def profile(req, address='localhost:8466'):
  from tensorflow.contrib.rpc.python.ops import rpc_op
  request = req.SerializeToString()
  op = rpc_op.rpc(protocol='grpc', method='/tensorflow.ProfilerService/Profile', request=request, address=address)
  return op

def parse_profile_result(res):
  response = profiler_service_pb2.ProfileResponse();
  response.ParseFromString(res)
  info = EasyDict(jf.MessageToDict(response))
  for entry in info.toolData:
    entry = EasyDict(entry)
    print(entry.name)
    json = b64decode(entry.data).decode('utf8')
    yield entry.name, json


def maketree(path):
    try:
        os.makedirs(path)
    except:
        pass


class Profile(threading.Thread):
  def __init__(self, req=None, **kws):
    if req is None:
      req = build_profile_request(**kws)
    else:
      assert len(kws) == 0
    super().__init__(daemon=True)
    self.req = req
    #with tf.Graph().as_default() as self.graph:
    self.op = profile(req)

  def run(self, session=None):
    if session is None:
      session = tf.compat.v1.get_default_session()
    self.path = 'tmp2/plugins/profile/{}_{}/'.format(int(time.time()), self.req.session_id)
    print('profiling for {}ms to {!r}'.format(self.req.duration_ms, self.path))
    self.data = self.op.eval(session=session)
    print('writing {} bytes to {!r}'.format(len(self.data), self.path))
    maketree(self.path)
    for name, json in parse_profile_result(self.data):
      tft.writebytes(self.path + name, json)
    return self.path
      
