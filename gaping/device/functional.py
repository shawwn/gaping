import enum
import math

import numpy as np

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import device_assignment as device_assignment_lib
from tensorflow.python.tpu import topology as topology_lib


class DeviceOrderMode(enum.IntEnum):
  """The way of determining device orders when computing device assignment."""
  # By default the mode is set to AUTO, the library will choose to form rings
  # when that is possible.
  AUTO = 0
  # Form rings for replicas and model-parallel cores.
  RING = 1
  # Form meshes for replicas and/or model-parallel cores.
  MESH = 2

def _ring_2d(height, width):
  """Ring-order of a height x width mesh.

  For example, in a 4x4 mesh, this returns the following order.
    0 -- 1 -- 2 -- 3
    |    |    |    |
    15-- 6 -- 5 -- 4
    |    |    |    |
    14-- 7 -- 8 -- 9
    |    |    |    |
    13-- 12-- 11-- 10

  Args:
    height: An integer represents the height.
    width: An integer represents the width.

  Returns:
    A list of [y, x] pairs with ring order.
  """
  if height == 1:
    return [(0, i) for i in range(width)]
  if width == 1:
    return [(i, 0) for i in range(height)]
  if height % 2 != 0:
    logging.warning("Odd dimension")
    return [(i % height, i // height) for i in range(width * height)]
  ret = [(0, 0)]
  for i in range(height // 2):
    for j in range(1, width):
      ret.append((2 * i, j))
    for j in range(width - 1, 0, -1):
      ret.append((2 * i + 1, j))
  for i in range(height - 1, 0, -1):
    ret.append((i, 0))
  return ret


def _open_ring_2d(x_size, y_size, z_coord):
  """Ring-order of a X by Y mesh, with a fixed Z coordinate.

  For example, in a 4x4 mesh, this returns the following order.
    0 -- 1 -- 2 -- 3
    |    |    |    |
    15-- 6 -- 5 -- 4
    |    |    |    |
    14-- 7 -- 8 -- 9
    |    |    |    |
    13-- 12-- 11-- 10

  Note that chip 0 is not included in the output.

  Args:
    x_size: An integer represents the mesh size in the x-dimension. Must be
      larger than 1.
    y_size: An integer represents the mesh size in the y-dimension. Must be
      larger than 1.
    z_coord: An integer represents the z-coordinate to use for the chips in the
      ring.

  Returns:
    A list of (x,y,z) triples in ring order.
  """
  ret = []
  for i in range(y_size // 2):
    for j in range(1, x_size):
      ret.append((j, 2 * i, z_coord))
    for j in range(x_size - 1, 0, -1):
      ret.append((j, 2 * i + 1, z_coord))
  for i in range(y_size - 1, 0, -1):
    ret.append((0, i, z_coord))
  return ret


def _ring_3d(x_size, y_size, z_size):
  """Ring-order of a X by Y by Z mesh.

  Constructs the 3d ring from 2d rings that are stacked in the Z dimension and
  joined in one corner.

  z == 0:
    0 -- 1 -- 2 -- 3
    |    |    |    |
    15 - 6 -- 5 -- 4
    |    |    |    |
    14 - 7 -- 8 -- 9
    |    |    |    |
    13 - 12 - 11 - 10
  z == 1:
    63 - 30 - 29 - 28
    |    |    |    |
    16 - 25 - 26 - 27
    |    |    |    |
    17 - 24 - 23 - 22
    |    |    |    |
    18 - 19 - 20 - 21
  z == 2:
    62 - 31 - 32 - 33
    |    |    |    |
    45 - 36 - 35 - 34
    |    |    |    |
    44 - 37 - 38 - 39
    |    |    |    |
    43 - 42 - 41 - 40
  z == 3:
    61 - 60 - 59 - 58
    |    |    |    |
    46 - 55 - 56 - 57
    |    |    |    |
    47 - 54 - 53 - 52
    |    |    |    |
    48 - 49 - 50 - 51

  Args:
    x_size: An integer represents the mesh size in the x-dimension. Must be
      larger than 1.
    y_size: An integer represents the mesh size in the y-dimension. Must be
      larger than 1.
    z_size: An integer represents the mesh size in the z-dimension. Must be
      larger than 1.  For example, in a 4x4x4 mesh, this returns the following
      order.

  Returns:
    A list of (x,y,z) triples in ring order.
  """
  # Handle the case where 2 dimensions are size 1.
  if x_size == 1 and y_size == 1:
    return [(0, 0, i) for i in range(z_size)]
  if x_size == 1 and z_size == 1:
    return [(0, i, 0) for i in range(y_size)]
  if y_size == 1 and z_size == 1:
    return [(i, 0, 0) for i in range(x_size)]
  # Handle odd mesh dimensions.  This never happens in practice, so we don't
  # bother to try building something optimal.
  if (x_size > 1 and x_size % 2 != 0) or (y_size > 1 and
                                          y_size % 2 != 0) or (z_size > 1 and
                                                               z_size % 2 != 0):
    logging.warning("Odd dimension")
    ret = []
    for z in range(z_size):
      for y in range(y_size):
        ret.extend((x, y, z) for x in range(x_size))
    return ret
  # Always start with chip 0.
  ret = [(0, 0, 0)]
  # Handle the case where one dimension is size 1.  We just build a flat, 2d
  # ring.
  if z_size == 1:
    ret.extend(_open_ring_2d(x_size, y_size, 0))
    return ret
  if y_size == 1:
    ret = [(0, 0, 0)]
    ret.extend((x, y, z) for (x, z, y) in _open_ring_2d(x_size, z_size, 0))
    return ret
  if x_size == 1:
    ret = [(0, 0, 0)]
    ret.extend((x, y, z) for (y, z, x) in _open_ring_2d(y_size, z_size, 0))
    return ret
  # Handle the case where all dimensions have size > 1 and even.
  ret = [(0, 0, 0)]
  for i in range(0, z_size):
    r = _open_ring_2d(x_size, y_size, i)
    if i % 2 == 0:
      ret.extend(r)
    else:
      ret.extend(reversed(r))
  for i in range(z_size - 1, 0, -1):
    ret.append((0, 0, i))
  return ret


def device_assignment_rank_3(topology,
                      computation_shape=None,
                      computation_stride=None,
                      num_replicas=1):
  """Computes a device_assignment of a computation across a TPU topology.

  Attempts to choose a compact grid of cores for locality.

  Returns a `DeviceAssignment` that describes the cores in the topology assigned
  to each core of each replica.

  `computation_shape` and `computation_stride` values should be powers of 2 for
  optimal packing.

  Args:
    topology: A `Topology` object that describes the TPU cluster topology.
      To obtain a TPU topology, evaluate the `Tensor` returned by
      `initialize_system` using `Session.run`. Either a serialized
      `TopologyProto` or a `Topology` object may be passed. Note: you must
      evaluate the `Tensor` first; you cannot pass an unevaluated `Tensor` here.
    computation_shape: A rank 1 int32 numpy array with size equal to the
      topology rank, describing the shape of the computation's block of cores.
      If None, the `computation_shape` is `[1] * topology_rank`.
    computation_stride: A rank 1 int32 numpy array of size `topology_rank`,
      describing the inter-core spacing of the `computation_shape` cores in the
      TPU topology. If None, the `computation_stride` is `[1] * topology_rank`.
    num_replicas: The number of computation replicas to run. The replicas will
      be packed into the free spaces of the topology.

  Returns:
    A DeviceAssignment object, which describes the mapping between the logical
    cores in each computation replica and the physical cores in the TPU
    topology.

  Raises:
    ValueError: If `topology` is not a valid `Topology` object.
    ValueError: If `computation_shape` or `computation_stride` are not 1D int32
      numpy arrays with shape [3] where all values are positive.
    ValueError: If computation's replicas cannot fit into the TPU topology.
  """
  # Deserialize the Topology proto, if it is a string.
  if isinstance(topology, bytes):
    topology = topology_lib.Topology(serialized=topology)

  if not isinstance(topology, topology_lib.Topology):
    raise ValueError("`topology` is not a Topology object; got {}".format(
        type(topology)))

  topology_rank = len(topology.mesh_shape)
  mesh_shape = topology.mesh_shape
  if computation_shape is None:
    computation_shape = np.array([1] * topology_rank, dtype=np.int32)
  else:
    computation_shape = np.asarray(computation_shape, dtype=np.int32)

  if computation_stride is None:
    computation_stride = np.array([1] * topology_rank, dtype=np.int32)
  else:
    computation_stride = np.asarray(computation_stride, dtype=np.int32)

  if computation_shape.shape != (topology_rank,):
    raise ValueError("computation_shape must have shape [{}]; got {}".format(
        topology_rank, computation_shape.shape))
  if computation_stride.shape != (topology_rank,):
    raise ValueError("computation_stride must have shape [{}]; got {}".format(
        topology_rank, computation_stride.shape))

  if any(computation_shape < 1):
    raise ValueError(
        "computation_shape must be positive; got computation_shape={}".format(
            computation_shape))
  if any(computation_stride < 1):
    raise ValueError(
        "computation_stride must be positive; got computation_stride={}".format(
            computation_stride))

  # Computes the physical size of one computation instance.
  computation_footprint = computation_shape * computation_stride
  if any(computation_footprint > mesh_shape):
    raise ValueError(
        "computation footprint {} does not fit in TPU topology shape {}".format(
            computation_footprint, mesh_shape))

  # Computes how many copies of the computation footprint fit in the mesh.
  block_counts = mesh_shape // computation_footprint

  replica_counts = block_counts * computation_stride
  max_replicas = np.prod(replica_counts)
  if num_replicas > max_replicas:
    raise ValueError(
        "requested {} replicas but only {} replicas with shape {} and "
        "computation_stride {} fit in a TPU mesh of shape {}".format(
            num_replicas, max_replicas, computation_shape, computation_stride,
            mesh_shape))

  def ceil_of_ratio(n, m):
    return (n + m - 1) // m

  replica_shape = [0] * topology_rank
  if num_replicas > 0:
    remaining_replicas = num_replicas
    remaining_dims = topology_rank

    # Choose dimensions as close to an equal cube as possible, in order of
    # increasing dimension size. By visiting dimensions in increasing size, we
    # assign the most constrained dimension first, so we won't make infeasible
    # choices.
    #
    # As a secondary sort order, visit the dimensions in reverse order. This
    # means we try to use both cores on the same chip in preference to two cores
    # on different chips.
    for x, ni in sorted(((x, -i) for (i, x) in enumerate(replica_counts))):
      i = -ni
      target_size = int(math.ceil(remaining_replicas**(1.0 / remaining_dims)))
      replica_shape[i] = min(target_size, x)
      remaining_replicas = ceil_of_ratio(remaining_replicas, replica_shape[i])
      remaining_dims -= 1

    assert remaining_replicas == 1 and remaining_dims == 0

  # Assigns an offset to each replica such that no two replicas overlap.
  replica_offsets = np.full([num_replicas, topology_rank], -1, dtype=np.int32)

  # TODO(ylc): Revisit here when topology_rank > 3.
  enable_2d_tiling = (
      topology_rank == 3 and
      computation_shape[-1] == 2  # Only handle 2D case.
      and np.prod(computation_stride) == 1  # Ensure no stride.
      and num_replicas == max_replicas)  # Full replication.
  logging.info("enable_2d_tiling: {}".format(enable_2d_tiling))
  if not enable_2d_tiling:
    logging.info("topology_rank: %s", topology_rank)
    logging.info("computation_shape: %s", computation_shape)
    logging.info("computation_stride: %s", computation_stride)
    logging.info("num_replicas: %s", num_replicas)
    logging.info("max_replicas: %s", max_replicas)
  if enable_2d_tiling:
    assignment = []
    inner_ring = _ring_2d(computation_shape[0], computation_shape[1])
    outer_ring = _ring_2d(replica_shape[0], replica_shape[1])

    for replica in range(num_replicas):
      outer_x, outer_y = outer_ring[replica]
      per_replica_assignment = []
      for index in range(np.prod(computation_shape)):
        inner_x, inner_y = inner_ring[index // 2]
        px = outer_x * computation_shape[0] + inner_x
        py = outer_y * computation_shape[1] + inner_y
        pz = index % 2
        per_replica_assignment.append([px, py, pz])
      assignment.append(per_replica_assignment)
  else:
    for replica in range(num_replicas):
      # Chooses a replica number in each axis.
      t = replica
      pos = []
      for dim in replica_shape[::-1]:
        pos.append(t % dim)
        t //= dim
      replica_pos = np.array(pos[::-1], dtype=np.int32)

      # Determines where that replica starts in each axis.
      outer = replica_pos // computation_stride
      inner = replica_pos % computation_stride
      replica_offsets[replica, :] = outer * computation_footprint + inner

    # Computes a logical core -> physical core mapping for each replica.
    indices = [
        np.arange(0, computation_shape[i] * computation_stride[i],
                  computation_stride[i]) for i in range(topology_rank)
    ]
    indices = np.concatenate(
        [i[..., np.newaxis] for i in np.meshgrid(*indices, indexing="ij")],
        axis=-1)
    indices = indices.reshape((-1, topology_rank))
    assignment = indices + replica_offsets[:, np.newaxis, :]

  return device_assignment_lib.DeviceAssignment(topology, core_assignment=assignment)


def device_assignment_rank_4(topology,
                      computation_shape=None,
                      computation_stride=None,
                      num_replicas=1,
                      device_order_mode=DeviceOrderMode.AUTO):
  """Computes a device_assignment of a computation across a TPU topology.

  Attempts to choose a compact grid of cores for locality.

  Returns a `DeviceAssignment` that describes the cores in the topology assigned
  to each core of each replica.

  `computation_shape` and `computation_stride` values should be powers of 2 for
  optimal packing.

  Args:
    topology: A `Topology` object that describes the TPU cluster topology.
      To obtain a TPU topology, evaluate the `Tensor` returned by
      `initialize_system` using `Session.run`. Either a serialized
      `TopologyProto` or a `Topology` object may be passed. Note: you must
      evaluate the `Tensor` first; you cannot pass an unevaluated `Tensor` here.
    computation_shape: A rank 1 int32 numpy array with size equal to the
      topology rank, describing the shape of the computation's block of cores.
      If None, the `computation_shape` is `[1] * topology_rank`.
    computation_stride: A rank 1 int32 numpy array of size `topology_rank`,
      describing the inter-core spacing of the `computation_shape` cores in the
      TPU topology. If None, the `computation_stride` is `[1] * topology_rank`.
    num_replicas: The number of computation replicas to run. The replicas will
      be packed into the free spaces of the topology.
    device_order_mode: An enum of `DeviceOrderMode` class which indicates
      whether to assign devices to form rings or meshes, or let the library to
      choose.

  Returns:
    A DeviceAssignment object, which describes the mapping between the logical
    cores in each computation replica and the physical cores in the TPU
    topology.

  Raises:
    ValueError: If `topology` is not a valid `Topology` object.
    ValueError: If `computation_shape` or `computation_stride` are not 1D int32
      numpy arrays with shape [3] where all values are positive.
    ValueError: If computation's replicas cannot fit into the TPU topology.
  """
  # Deserialize the Topology proto, if it is a string.
  if isinstance(topology, bytes):
    topology = topology_lib.Topology(serialized=topology)

  if not isinstance(topology, topology_lib.Topology):
    raise ValueError("`topology` is not a Topology object; got {}".format(
        type(topology)))

  topology_rank = len(topology.mesh_shape)
  mesh_shape = topology.mesh_shape
  if computation_shape is None:
    computation_shape = np.array([1] * topology_rank, dtype=np.int32)
  else:
    computation_shape = np.asarray(computation_shape, dtype=np.int32)

  if computation_stride is None:
    computation_stride = np.array([1] * topology_rank, dtype=np.int32)
  else:
    computation_stride = np.asarray(computation_stride, dtype=np.int32)

  if computation_shape.shape != (topology_rank,):
    raise ValueError("computation_shape must have shape [{}]; got {}".format(
        topology_rank, computation_shape.shape))
  if computation_stride.shape != (topology_rank,):
    raise ValueError("computation_stride must have shape [{}]; got {}".format(
        topology_rank, computation_stride.shape))

  if any(computation_shape < 1):
    raise ValueError(
        "computation_shape must be positive; got computation_shape={}".format(
            computation_shape))
  if any(computation_stride < 1):
    raise ValueError(
        "computation_stride must be positive; got computation_stride={}".format(
            computation_stride))

  # Computes the physical size of one computation instance.
  computation_footprint = computation_shape * computation_stride
  if any(computation_footprint > mesh_shape):
    raise ValueError(
        "computation footprint {} does not fit in TPU topology shape {}".format(
            computation_footprint, mesh_shape))

  # Computes how many copies of the computation footprint fit in the mesh.
  block_counts = mesh_shape // computation_footprint

  replica_counts = block_counts * computation_stride
  max_replicas = np.prod(replica_counts)
  if num_replicas > max_replicas:
    raise ValueError(
        "requested {} replicas but only {} replicas with shape {} and "
        "computation_stride {} fit in a TPU mesh of shape {}".format(
            num_replicas, max_replicas, computation_shape, computation_stride,
            mesh_shape))

  def ceil_of_ratio(n, m):
    return (n + m - 1) // m

  if topology.missing_devices.size == 0:
    replica_shape = [0] * topology_rank
    if num_replicas > 0:
      remaining_replicas = num_replicas
      remaining_dims = topology_rank

      # Choose dimensions as close to an equal cube as possible,
      # in order of increasing dimension size. By visiting dimensions
      # in increasing size, we assign the most constrained dimension
      # first, so we won't make infeasible choices.
      #
      # As a secondary sort order, visit the last dimension (core index) first,
      # then the other dimensions in increasing order. This means we try to use
      # both cores on the same chip in preference to two cores on different
      # chips.  We visit the x dimension first, and the z dimension last, so
      # that we prefer to arrange adjacent replicas on the same machine when
      # possible.
      #
      # For example, if num_replicas == 4, we prefer to use a replica_shape of
      # (2,1,1,2) over (1,1,2,2).

      for x, ni in sorted(((x, ((i + 1) % topology_rank))
                           for (i, x) in enumerate(replica_counts))):
        i = (ni + topology_rank - 1) % topology_rank
        target_size = int(math.ceil(remaining_replicas**(1.0 / remaining_dims)))
        replica_shape[i] = min(target_size, x)
        remaining_replicas = ceil_of_ratio(remaining_replicas, replica_shape[i])
        remaining_dims -= 1

      assert remaining_replicas == 1 and remaining_dims == 0

    # Assigns an offset to each replica such that no two replicas overlap.
    replica_offsets = np.full([num_replicas, topology_rank], -1, dtype=np.int32)

    enable_3d_tiling = (
        topology_rank == 4 and
        computation_shape[-1] == 2  # Only handle 3D case.
        and np.prod(computation_stride) == 1  # Ensure no stride.
        and num_replicas == max_replicas)  # Full replication.

    if device_order_mode != DeviceOrderMode.AUTO:
      if device_order_mode == DeviceOrderMode.RING and not enable_3d_tiling:
        raise ValueError("cannot assign ring order in the given topology")
      enable_3d_tiling = device_order_mode == DeviceOrderMode.RING
    logging.info("enable_3d_tiling: {}".format(enable_3d_tiling))
    if not enable_3d_tiling:
      logging.info("topology_rank: %s", topology_rank)
      logging.info("computation_shape: %s", computation_shape)
      logging.info("computation_stride: %s", computation_stride)
      logging.info("num_replicas: %s", num_replicas)
      logging.info("max_replicas: %s", max_replicas)


    if enable_3d_tiling:
      assignment = []
      inner_ring = _ring_3d(computation_shape[0], computation_shape[1],
                            computation_shape[2])
      outer_ring = _ring_3d(replica_shape[0], replica_shape[1],
                            replica_shape[2])

      for replica in range(num_replicas):
        outer_x, outer_y, outer_z = outer_ring[replica]
        per_replica_assignment = []
        for index in range(np.prod(computation_shape)):
          inner_x, inner_y, inner_z = inner_ring[index // 2]
          px = outer_x * computation_shape[0] + inner_x
          py = outer_y * computation_shape[1] + inner_y
          pz = outer_z * computation_shape[2] + inner_z
          pi = index % 2
          per_replica_assignment.append([px, py, pz, pi])
        assignment.append(per_replica_assignment)
    else:
      for replica in range(num_replicas):
        # Chooses a replica number in each axis.
        t = replica
        pos = []
        # Visit the core number first.
        for dim in np.concatenate([[replica_shape[-1]], replica_shape[:-1]]):
          pos.append(t % dim)
          t //= dim
        replica_pos = np.concatenate([pos[1:], [pos[0]]])

        # Determines where that replica starts in each axis.
        outer = replica_pos // computation_stride
        inner = replica_pos % computation_stride
        replica_offsets[replica, :] = outer * computation_footprint + inner

      # Computes a logical core -> physical core mapping for each replica.
      indices = [
          np.arange(0, computation_shape[i] * computation_stride[i],
                    computation_stride[i]) for i in range(topology_rank)
      ]
      indices = np.concatenate(
          [i[..., np.newaxis] for i in np.meshgrid(*indices, indexing="ij")],
          axis=-1)
      indices = indices.reshape((-1, topology_rank))
      assignment = indices + replica_offsets[:, np.newaxis, :]
  else:
    # We have a slice with missing chips. We define a simple assignment by
    # ignoring computation stride. This assignment should enable a consistent
    # and correct device assignment on degraded slices. It is optimal when
    # weights are not sharded. But this device assignment may be sub-optimal for
    # other model parallelism scenarios.
    assert np.prod(computation_stride) == 1
    # Next, we check if we have sufficient devices.
    assert num_replicas * np.prod(
        computation_shape) <= topology.num_tasks * topology.num_tpus_per_task
    # Map replicas to physical devices in task order.
    device_coordinates = topology.device_coordinates
    assignment = []
    devices_per_replica = np.prod(computation_shape)
    for rindex in range(num_replicas):
      replica_assignment = []
      for index in range(devices_per_replica):
        logical_id = rindex * devices_per_replica + index
        # Pick logical cores in task order
        task = logical_id // topology.num_tpus_per_task
        device = logical_id % topology.num_tpus_per_task
        # Append physical cores to the replica assignment
        replica_assignment.append(device_coordinates[task, device, :])
      assignment.append(replica_assignment)

  return device_assignment_lib.DeviceAssignment(topology, core_assignment=assignment)


def device_assignment(topology,
                      computation_shape=None,
                      computation_stride=None,
                      num_replicas=1,
                      device_order_mode=DeviceOrderMode.AUTO):
  """Computes a device_assignment of a computation across a TPU topology.

  Attempts to choose a compact grid of cores for locality.

  Returns a `DeviceAssignment` that describes the cores in the topology assigned
  to each core of each replica.

  `computation_shape` and `computation_stride` values should be powers of 2 for
  optimal packing.

  Args:
    topology: A `Topology` object that describes the TPU cluster topology.
      To obtain a TPU topology, evaluate the `Tensor` returned by
      `initialize_system` using `Session.run`. Either a serialized
      `TopologyProto` or a `Topology` object may be passed. Note: you must
      evaluate the `Tensor` first; you cannot pass an unevaluated `Tensor` here.
    computation_shape: A rank 1 int32 numpy array with size equal to the
      topology rank, describing the shape of the computation's block of cores.
      If None, the `computation_shape` is `[1] * topology_rank`.
    computation_stride: A rank 1 int32 numpy array of size `topology_rank`,
      describing the inter-core spacing of the `computation_shape` cores in the
      TPU topology. If None, the `computation_stride` is `[1] * topology_rank`.
    num_replicas: The number of computation replicas to run. The replicas will
      be packed into the free spaces of the topology.
    device_order_mode: An enum of `DeviceOrderMode` class which indicates
      whether to assign devices to form rings or meshes, or let the library to
      choose.

  Returns:
    A DeviceAssignment object, which describes the mapping between the logical
    cores in each computation replica and the physical cores in the TPU
    topology.

  Raises:
    ValueError: If `topology` is not a valid `Topology` object.
    ValueError: If `computation_shape` or `computation_stride` are not 1D int32
      numpy arrays with shape [3] where all values are positive.
    ValueError: If computation's replicas cannot fit into the TPU topology.
  """
  topology_rank = get_topology_rank(topology)
  computation_shape = adjust_computation_shape(computation_shape, topology)
  computation_stride = adjust_computation_shape(computation_stride, topology)
  if topology_rank == 3:
    return device_assignment_rank_3(topology,
        computation_shape=computation_shape,
        computation_stride=computation_stride,
        num_replicas=num_replicas)
  if topology_rank == 4:
    return device_assignment_rank_4(topology,
        computation_shape=computation_shape,
        computation_stride=computation_stride,
        num_replicas=num_replicas,
        device_order_mode=device_order_mode)
  raise ValueError("Unexpected topology rank %d" % topology_rank)


def adjust_computation_shape(shape, topology):
  topology_rank = get_topology_rank(topology)
  if shape is None:
    return np.array([1] * topology_rank, dtype=np.int32)
  computation_rank = len(shape)
  if computation_rank == topology_rank:
    return np.array(shape, dtype=np.int32)
  if computation_rank == 3 and topology_rank == 4:
    return np.array([shape[0], shape[1], 1, shape[2]], dtype=np.int32)
  if computation_rank == 4 and topology_rank == 3:
    if shape[2] != 1:
      raise ValueError("Expected computation shape index 2 to be 1")
    return np.array([shape[0], shape[1], shape[3]], dtype=np.int32)
  raise ValueError("Unexpected topology rank %d vs computation rank %d" % (topology_rank, computation_rank))


class ComputationInfo(dict):
  def __init__(self, topology=None, computation_shape=None, computation_stride=None):
    if topology is not None:
      get_computation_info(topology=topology, computation_shape=computation_shape, computation_stride=computation_stride, info=self)
  def __getattr__(self, k):
    try:
      return self[k]
    except KeyError:
      raise AttributeError(k)
  def __setattr__(self, k, v):
    self[k] = v
  def __delattr__(self, k):
    del self[k]

def get_computation_info(topology, computation_shape=None, computation_stride=None, info=None):
  """Computes the physical size of one computation instance."""
  computation_shape = adjust_computation_shape(computation_shape, topology)
  computation_stride = adjust_computation_shape(computation_stride, topology)
  mesh_shape = topology.mesh_shape
  computation_footprint = computation_shape * computation_stride
  if any(computation_footprint > mesh_shape):
    raise ValueError(
        "computation footprint {} does not fit in TPU topology shape {}".format(
            computation_footprint, mesh_shape))
  # Computes how many copies of the computation footprint fit in the mesh.
  block_counts = mesh_shape // computation_footprint
  replica_counts = block_counts * computation_stride
  max_blocks = np.prod(block_counts)
  max_replicas = np.prod(replica_counts)
  if info is None:
    info = ComputationInfo()
  info.mesh_shape = mesh_shape
  info.computation_shape = computation_shape
  info.computation_stride = computation_stride
  info.computation_footprint = computation_footprint
  info.block_counts = block_counts
  info.replica_counts = replica_counts
  info.max_blocks = max_blocks
  info.max_replicas = max_replicas
  return info

def get_computation_replica_shape(info, num_replicas=None):
  topology_rank = len(info.mesh_shape)
  replica_counts = info.replica_counts
  replica_shape = [0] * topology_rank
  if num_replicas is None:
    num_replicas = info.max_replicas
  if num_replicas > 0:
    remaining_replicas = num_replicas
    remaining_dims = topology_rank
    # Choose dimensions as close to an equal cube as possible, in order of
    # increasing dimension size. By visiting dimensions in increasing size, we
    # assign the most constrained dimension first, so we won't make infeasible
    # choices.
    #
    # As a secondary sort order, visit the dimensions in reverse order. This
    # means we try to use both cores on the same chip in preference to two cores
    # on different chips.
    for x, ni in sorted(((x, -i) for (i, x) in enumerate(replica_counts))):
      i = -ni
      target_size = int(math.ceil(remaining_replicas**(1.0 / remaining_dims)))
      replica_shape[i] = min(target_size, x)
      remaining_replicas = ceil_of_ratio(remaining_replicas, replica_shape[i])
      remaining_dims -= 1
    assert remaining_replicas == 1 and remaining_dims == 0
  return replica_shape

def ceil_of_ratio(n, m):
  return (n + m - 1) // m

def get_topology_rank(topology):
  return len(topology.mesh_shape)
