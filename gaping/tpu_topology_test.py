import tensorflow as tf
import tensorflow.compat.v1 as tf1
from tensorflow.python.tpu import topology as topology_lib

from absl import flags
from absl.testing import parameterized

from gaping import test_utils
import base64
import numpy as np

from tensorflow.core.protobuf.tpu import topology_pb2

from google.protobuf.json_format import MessageToJson
import json

def pb_to_json(pb):
  return json.loads(MessageToJson(pb))


TPU_TOPOLOGIES = [base64.b64decode(x) for x in [
    "CgQCAgECEAEYCCIgAAAAAAAAAAEBAAAAAQAAAQABAAAAAQABAQEAAAEBAAE=",
    "CgMCAgIQARgIIhgAAAAAAAEAAQAAAQEBAAABAAEBAQABAQE=",
    ]]


class TpuTopologyTest(parameterized.TestCase, test_utils.GapingTestCase):

  def setUp(self):
    super().setUp()

  @parameterized.parameters(TPU_TOPOLOGIES)
  def testTpuTopology(self, serialized):
    proto = topology_pb2.TopologyProto()
    proto.ParseFromString(serialized)
    mesh_shape = np.array(proto.mesh_shape, dtype=np.int32)
    self.log(pb_to_json(proto))
    if proto.num_tasks < 0:
      raise ValueError("`num_tasks` must be >= 0; got {}".format(proto.num_tasks))
    if proto.num_tpu_devices_per_task < 0:
      raise ValueError("`num_tpu_devices_per_task` must be >= 0; got {}".format(proto.num_tpu_devices_per_task))
    expected_coordinates_size = (proto.num_tasks * proto.num_tpu_devices_per_task * len(proto.mesh_shape))
    if len(proto.device_coordinates) != expected_coordinates_size:
      raise ValueError("`device_coordinates` must have shape num_tasks ({}) * "
                       "num_tpu_devices_per_task ({}) * len(mesh_shape) ({}); "
                       "got shape {}".format(proto.num_tasks,
                                             proto.num_tpu_devices_per_task,
                                             proto.mesh_shape,
                                             len(proto.device_coordinates)))
    coords = np.array(proto.device_coordinates, dtype=np.int32)
    if any(coords < 0):
      raise ValueError("`device_coordinates` must be >= 0")
    coords = coords.reshape((proto.num_tasks, proto.num_tpu_devices_per_task, len(proto.mesh_shape)))
    self.log(coords)
    if len(proto.device_coordinates) != expected_coordinates_size:
      raise ValueError("`device_coordinates` must have shape num_tasks ({}) * "
                       "num_tpu_devices_per_task ({}) * len(mesh_shape) ({}); "
                       "got shape {}".format(proto.num_tasks,
                                             proto.num_tpu_devices_per_task,
                                             proto.mesh_shape,
                                             len(proto.device_coordinates)))
    
  @parameterized.parameters(TPU_TOPOLOGIES)
  def testTpuTopologyObject(self, serialized):
    topology = topology_lib.Topology(serialized=serialized)
    tasks, devices = topology._invert_topology()
    self.log('tasks   %s %s', tasks.shape, tasks)
    self.log('devices %s %s', devices.shape, devices)
    
    
    

if __name__ == "__main__":
  import gaping.wrapper
  with gaping.wrapper.patch_tensorflow():
    tf.test.main()

