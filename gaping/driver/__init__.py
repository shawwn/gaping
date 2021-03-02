from .. import wrapper
from .. import tf_tools as tft

import tensorflow as tf
import time

from tensorflow.python.framework import ops
from tensorflow.python.ops import critical_section_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.tpu import topology as topology_lib

from contextlib import contextmanager
import os

@contextmanager
def use_session(session):
  with session.graph.as_default(), session.as_default():
    yield session


@contextmanager
def fork_session(session, graph=None):
  if graph is None:
    graph = tf.Graph()
  elif graph is True:
    graph = session.graph
  new_session = wrapper.clone_session(session=session, graph=graph)
  with new_session:
    with use_session(new_session) as sess:
      yield sess


class Driver:
  def __init__(self, session):
    self.session = session
    self.session.driver = getattr(session, 'driver', self)

  @property
  def graph(self):
    return self.session.graph

  def run(self, *args, **kws):
    return self.session.run(*args, **kws)

  def use(self):
    return use_session(self.session)

  def fork(self, graph=None):
    return fork_session(self.session, graph=graph)

  def close(self):
    self.session.close()

  def uniq(self, base='x'):
    with self.use():
      return base + str(int(time.time() * 1e9)) + '_' + str(ops.uid())


class CreateSessionDriver(Driver):
  def __init__(self, target='', graph=None, config=None, interactive=False):
    if config is None:
      config = wrapper.make_session_config()
    if graph is None:
      graph = tf.compat.v1.get_default_graph()
    Session = tf.compat.v1.InteractiveSession if interactive else tf.compat.v1.Session
    session = Session(target, graph=graph, config=config)
    super().__init__(session=session)

  
class CPUDriver(CreateSessionDriver):
  def __init__(self, graph=None, interactive=False):
    super().__init__(graph=graph, interactive=interactive)


def localvar(name, dtype, shape=(), shared=True, initial_value=None):
  with ops.init_scope(), tf.variable_scope('', reuse=tf.AUTO_REUSE):
    if not shared:
      name = get().uniq(name)
    if initial_value is None:
      initializer = tf.initializers.zeros
    else:
      if dtype is None:
        import pdb; pdb.set_trace()
      initializer = tf.initializers.constant(value=initial_value, dtype=dtype)
    return tf.get_variable(name,
        dtype=dtype, shape=shape, initializer=initializer,
        collections=['local_variables'], use_resource=True, trainable=False)


def get_tpu_topology_var():
  return localvar('tpu_topology', tf.string)


def fetch_tpu_topology_unsafe(tpu_topology_var):
  with ops.colocate_with(tpu_topology_var):
    return tf.cond(state_ops.is_variable_initialized(tpu_topology_var),
        lambda: [False, tpu_topology_var.read_value()],
        lambda: [True, tpu_topology_var.assign(tf.tpu.initialize_system())])


def fetch_tpu_topology(tpu_topology_var=None):
  if tpu_topology_var is None:
    tpu_topology_var = get_tpu_topology_var()
  # this doesn't actually protect against multiple processes
  # initializing a TPU simultaneously. I think it has to do with the
  # fact that tf.tpu.initialize_system() is on /device:TPU_SYSTEM:0
  # whereas the tpu_topology_var is on the TPU CPU.
  # cs = critical_section_ops.CriticalSection(name="tpu_topology_lock", shared_name="tpu_topology_lock")
  # return cs.execute(lambda: fetch_tpu_topology_unsafe(tpu_topology_var))
  return fetch_tpu_topology_unsafe(tpu_topology_var)


def get_tpu_topology(serialized=None, session=None):
  if session is None:
    session = tf.get_default_session()
  if serialized is None:
    with fork_session(session) as sess: # don't pollute the existing graph
      initialized, serialized = sess.run(fetch_tpu_topology())
      tf.logging.info('%s %s',
        'Initialized TPU' if initialized else "Did not initialize TPU",
        sess._target.decode('utf-8'))
  topology = topology_lib.Topology(serialized=serialized)
  return topology


class TPUDriver(CreateSessionDriver):
  def __init__(self, tpu=None, zone=None, project=None, graph=None, interactive=False, topology=None):
    self.resolver = wrapper.TPUClusterResolver(tpu=tpu, zone=zone, project=project)
    target = self.resolver.get_master()
    cluster_spec = self.resolver.cluster_spec()
    cluster_def = cluster_spec.as_cluster_def()
    tf.logging.info('%s', cluster_def)
    config = wrapper.make_session_config(cluster_spec=cluster_spec)
    super().__init__(target=target, graph=graph, config=config, interactive=interactive)
    self._topology = None

  @property
  def topology(self):
    if self._topology is None:
      self._topology = get_tpu_topology(session=self.session)
    return self._topology

  def device_assignment(self, cores=None):
    if cores is not None:
      return wrapper.get_core_assignment( cores, topology=self.topology )
    else:
      return wrapper.get_device_assignment( topology=self.topology )

  def shard(self, fn, device_assignment=None, *args, **kws):
    if device_assignment is None:
      device_assignment = self.device_assignment()
    return tft.tpu_shard(fn, device_assignment=device_assignment, *args, **kws)


def new(tpu=None, **kws):
  zone = kws.pop('zone', None)
  project = kws.pop('project', None)
  if tpu is None:
    tpu = os.environ.get('TPU_NAME')
  if tpu:
    return TPUDriver(tpu=tpu, zone=zone, project=project, **kws)
  else:
    return CPUDriver(**kws)


def get(session=None):
  if session is None:
    session = tf.compat.v1.get_default_session()
  return getattr(session, 'driver', None)
