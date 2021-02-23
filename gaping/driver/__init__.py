from .. import wrapper

import tensorflow as tf

from contextlib import contextmanager
import os

class Driver:
  def __init__(self, session):
    self.session = session

  @property
  def graph(self):
    return self.session.graph

  def run(self, *args, **kws):
    return self.session.run(*args, **kws)

  @contextmanager
  def use(self):
    with self.graph.as_default(), self.session.as_default():
      yield

  @contextmanager
  def fork(self, graph=None):
    if graph is None:
      graph = tf.Graph()
    elif graph is True:
      graph = self.graph
    session = wrapper.clone_session(session=self.session, graph=graph)
    with session, self.graph.as_default(), self.session.as_default():
      yield session


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


from tensorflow.python.tpu import tpu_strategy_util
from tensorflow.python.tpu import topology as topology_lib
from tensorflow.python.framework import errors_impl


def fetch_tpu_topology(resolver, session=None):
  if session is None:
    session = tf.get_default_session()
  with Driver(session).fork(): # don't pollute the existing graph
    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
      tpu_topology_var = tf.get_variable('tpu_topology',
          dtype=tf.string, shape=(), initializer=tf.initializers.zeros,
          collections=['local_variables'], use_resource=True, trainable=False)
    try:
      serialized = tpu_topology_var.eval()
      topology = topology_lib.Topology(serialized=serialized)
    except errors_impl.FailedPreconditionError: # variable doesn't exist
      topology = tpu_strategy_util.initialize_tpu_system(resolver)
      tpu_topology_var.load(topology.serialized())
    return topology


class TPUDriver(CreateSessionDriver):
  def __init__(self, tpu=None, zone=None, project=None, graph=None, interactive=False):
    self.resolver = wrapper.TPUClusterResolver(tpu=tpu, zone=zone, project=project)
    target = self.resolver.get_master()
    cluster_spec = self.resolver.cluster_spec()
    cluster_def = cluster_spec.as_cluster_def()
    tf.logging.info('%s', cluster_def)
    config = wrapper.make_session_config(cluster_spec=cluster_spec)
    super().__init__(target=target, graph=graph, interactive=interactive)
    self.topology = fetch_tpu_topology(resolver=self.resolver, session=self.session)


def new(tpu=None, **kws):
  zone = kws.pop('zone', None)
  project = kws.pop('project', None)
  if tpu is None:
    tpu = os.environ.get('TPU_NAME')
  if tpu:
    return TPUDriver(tpu=tpu, zone=zone, project=project, **kws)
  else:
    return CPUDriver(**kws)
