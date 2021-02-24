from .. import driver as driver_lib
from .. import tf_tools as tft

import tensorflow as tf

from . import abstract_sequencer
from . import event_sequencer
from . import event_sink
from . import cursored
from . import sequence
from . import util
from . import initializable
from ..util import EasyDict


class RingBufferFields(initializable.Initializable):
  def __init__(self, event_factory, sequencer):
    self.sequencer = sequencer
    self.buffer_size = sequencer.get_buffer_size()
    self.buffer_size = util.check(self.buffer_size, self.buffer_size >= 1,
        "buffer_size must not be less than 1")
    # self.buffer_size = util.check(self.buffer_size, util.bit_count(self.buffer_size) == 1,
    #     "buffer_size must be a power of 2")

    # this.indexMask = bufferSize - 1;
    # this.entries = new Object[sequencer.getBufferSize() + 2 * BUFFER_PAD];
    # fill(eventFactory);
    self.infeed = None
    self.outfeed = None
    self.fill(event_factory)

  def get_initializers(self):
    return super().get_initializers() + \
        self.sequencer.get_initializers()

  def fill(self, event_factory):
    # for (int i = 0; i < bufferSize; i++)
    # {
    #     entries[BUFFER_PAD + i] = eventFactory.newInstance();
    # }
    self.infeed, self.outfeed = event_factory(self.buffer_size)
  
  def element_at(self, feed, sequence):
    #return feed.peek(sequence)
    key, value = feed.get(sequence) # removes from feed
    key.set_shape([])
    result = EasyDict(value)
    result['index'] = key
    return result
  
  def element_set(self, feed, sequence, event):
    return feed.put(sequence, event)
    

class RingBuffer(
    RingBufferFields,
    event_sequencer.EventSequencer,
    event_sink.EventSink,
    cursored.Cursored):

  INITIAL_CURSOR_VALUE = sequence.Sequence.INITIAL_VALUE

  def __init__(self, event_factory, sequencer):
    super().__init__(event_factory, sequencer)

  @classmethod
  def create_single_producer(cls, factory, buffer_size, wait_strategy=None):
    # if wait_strategy is None:
    #   wait_strategy = BlockingWaitStrategy()
    sequencer = abstract_sequencer.SingleProducerSequencer(buffer_size, wait_strategy)
    return cls(factory, sequencer)

  def iget(self, sequence=None):
    return self.element_at(self.infeed, sequence)

  def iput(self, event, sequence=None):
    if sequence is None:
      sequence = tft.tpu_nanos()
    return self.element_set(self.infeed, sequence, event)

  def get(self, sequence=None):
    return self.element_at(self.outfeed, sequence)

  def put(self, event, sequence=None):
    return self.element_set(self.outfeed, sequence, event)

  def next_n(self, n):
    return self.sequencer.next_n(n)

  def try_next_n(self, n):
    return self.sequencer.try_next_n(n)

  def reset_to(self, sequence):
    with util.dep(self.sequencer.claim(sequence)):
      return self.sequencer.publish(sequence)

  def is_published(self, sequence):
    return self.sequencer.is_available(sequence)

  def add_gating_sequences(self, *gating_sequences):
    return self.sequencer.add_gating_sequences(*gating_sequences)

  def get_minimum_gating_sequence(self):
    return self.sequencer.get_minimum_sequence()

  def remove_gating_sequence(self, sequence):
    return self.sequencer.remove_gating_sequence(sequence)

  def new_barrier(self, *sequences_to_track):
    return self.sequencer.new_barrier(sequences_to_track)

  def new_poller(self, *gating_sequences):
    return self.sequencer.new_poller(self, gating_sequences)

  def get_cursor(self):
    return self.sequencer.get_cursor()

  def get_buffer_size(self):
    return self.buffer_size

  def has_available_capacity(self, required_capacity):
    return self.sequencer.has_available_capacity(required_capacity)

  def remaining_capacity(self):
    return self.sequencer.remaining_capacity()

  def publish_event(self, translator, *args, **kws):
    sequence = self.sequencer.next()
    return self.translate_and_publish(translator, sequence, *args, **kws)

  def try_publish_event(self, translator, *args, **kws):
    sequence = self.sequencer.try_next()
    def ok():
      with util.dep(self.translate_and_publish(translator, sequence, *args, **kws)):
        return tf.constant(True)
    def fail():
      return tf.constant(False)
    return tf.cond(sequence >= 0, ok, fail)

  def translate_and_publish(self, translator, sequence, *args, **kws):
    #event = self.iget(sequence)
    event = self.iget()
    event = translator(event, sequence, *args, **kws)
    with util.dep(self.put(event, sequence=sequence)):
      return self.sequencer.publish(sequence)

  def publish(self, sequence):
    return self.sequencer.publish(sequence)

  def publish_range(self, lo, hi):
    return self.sequencer.publish_range(lo, hi)

  def __str__(self):
    return type(self).__name__ + "{" + \
        "buffer_size=" + str(self.buffer_size) + \
        ", sequencer=" + str(self.sequencer) + \
        "}"

  def __repr__(self):
    return str(self)


test = """

from gaping.util import EasyDict

from gaping.driver import util; reload(util)

BUFFER_SIZE = 16

from gaping.driver import abstract_sequencer; reload(abstract_sequencer); seqsr = abstract_sequencer.SingleProducerSequencer(BUFFER_SIZE, None); r( seqsr.initializer )

from gaping.driver import pipe; reload(pipe); infeed_builder = pipe.PipeBuilder(); outfeed_builder = pipe.PipeBuilder(); hostfeed_builder = pipe.PipeBuilder()

infeed_builder.channel('label', tf.int32); infeed_builder.channel('image', tf.string); outfeed_builder.channel('label', tf.float32, shape=[1000]); outfeed_builder.channel('image', tf.float32, shape=[16,16,3]); hostfeed_builder.channel('label', tf.int64); hostfeed_builder.channel('image', tf.string); #hostfeed_builder.channel('index', tf.int64);

# >>> rb.outfeed.name
# 'pipe1614125718836599040_1701'
# >>> rb.infeed.name
# 'pipe1614125718836194048_1700'

infeed = infeed_builder.get(shared_name = 'pipe1614125718836194048_1700')
outfeed = outfeed_builder.get(shared_name = 'pipe1614125718836599040_1701')
hostfeed = hostfeed_builder.get(shared_name='pipe1614129327201607936_3027')

from gaping.driver import ring_buffer; reload(ring_buffer); rb = ring_buffer.RingBuffer(lambda buffer_size: (infeed, outfeed), seqsr)

def rand_label(n=1000):
  return tf.random.uniform(minval=0, maxval=n, dtype=tf.int32, shape=())

def rand_jpeg(width=16, height=16):
  return tf.io.encode_jpeg(tf.cast(255*tf.random.uniform(shape=[width, height, 3]), tf.uint8))

def parse_jpeg(e, seq=None):
  return EasyDict(
    label=tf.one_hot(e.label, 1000),
    image=tft.with_shape([16,16,3], lambda: tft.transform_image(e.image)))

def infeeder(i=None):
  return rb.iput(EasyDict(label=rand_label(), image=rand_jpeg()))

def outfeeder(i=None):
  return rb.try_publish_event(parse_jpeg)

def pumpfeed(i=None):
  with util.dep(infeeder()):
    return outfeeder()

pump10 = util.loop_reduce(pumpfeed, 10)
pump100 = util.loop_reduce(pumpfeed, 100)
pump1000 = util.loop_reduce(pumpfeed, 1000)
pump10000 = util.loop_reduce(pumpfeed, 10000)
pump100000 = util.loop_reduce(pumpfeed, 100000)

osize = rb.outfeed.size()

import time

while True: now = time.time(); pump10000.eval(); elapsed = (time.time() - now); olen = osize.eval(); elapsed, 10000/elapsed, olen, olen*16*16*3*4/1024/1024/1024


def encode_jpeg(image, **kws):
  def on_cpu(x):
    return tf.io.encode_jpeg(x, **kws)
  if image.dtype in [tf.float32, tf.float64]:
    image = tf.cast(255*image, tf.uint8)
  return tft.tpu_cpu(on_cpu, image)

def encode_host(e, seq=None):
  return EasyDict(
      label=tf.argmax(e.label, axis=-1),
      image=encode_jpeg(e.image))


def sharded_inputs(getter, shards=8):
  values = [getter() for _ in range(shards)]
  inputs = [tf.stack(x) for x in zip(*[tf.nest.flatten(values[i]) for i in range(shards)])]
  def unpack(args):
    args = [tf.squeeze(x, axis=0) for x in args]
    info = tf.nest.pack_sequence_as(values[0], args)
    return info
  return inputs, unpack

#inputs, unpacker = sharded_inputs(rb.get)


def prn(x, *args):
  print(x, *args)
  return x


#op = shard(lambda *args: prn(unpacker(args)) and [tf.constant(0)], inputs=inputs)

from gaping.driver.padded_long import make_mutex

from tensorflow.contrib import tpu

def sharded_pipeline(outfeed, getter, encoder, shards=8, iterations=1):
  mutex = make_mutex('pipeline_mutex')
  inputs, unpacker = sharded_inputs(getter, shards=shards)
  def tpu_step(*args):
    e = unpacker(args)
    del e.index
    def on_cpu():
      o = encoder(e)
      prn(o)
      with util.dep(outfeed.put(tft.tpu_nanos(), o)):
        return outfeed.size()
    return tft.tpu_cpu(on_cpu, mutex=mutex)
  return shard(tpu_step, inputs=inputs)

from gaping import driver as driver_lib

def get_num_replicas():
  return driver_lib.get().device_assignment().num_replicas

def sharded_pipeline(outfeed, infeed, encoder, iterations=1, device_assignment=None):
  if device_assignment is None:
    device_assignment = driver_lib.get().device_assignment()
  mutex = make_mutex('pipeline_mutex')
  def tpu_step(i=None):
    def on_cpu():
      e = EasyDict(infeed.get()[1])
      o = encoder(e)
      prn(o)
      # with util.dep(outfeed.put(tft.tpu_nanos(), o)):
      #   return outfeed.size()
      with util.dep(outfeed.put(tft.tpu_nanos(), o)):
        return tf.constant(0)
    return tft.tpu_cpu(on_cpu, mutex=mutex)
  def tpu_iterations():
    return tf.minimum(infeed.size() // 8, iterations)
  def tpu_loop(n):
    n = n[0]
    #n = tft.tpu_cpu(tpu_iterations)
    prn(n)
    #ret = tpu.repeat(n, tpu_step, [util.ti32(0)])
    ret = util.loop_reduce(tpu_step, n)
    prn(ret)
    return ret
  inputs = [[tpu_iterations() for _ in range(device_assignment.num_replicas)]]
  was = infeed.size()
  result = driver_lib.get().shard(tpu_loop, inputs=inputs, device_assignment=device_assignment)
  with tf.control_dependencies(result):
    now = infeed.size()
    return was - now, now

op = sharded_pipeline(hostfeed, rb.outfeed, encode_host, iterations=1)
op10 = sharded_pipeline(hostfeed, rb.outfeed, encode_host, iterations=10)
op100 = sharded_pipeline(hostfeed, rb.outfeed, encode_host, iterations=100)
op1000 = sharded_pipeline(hostfeed, rb.outfeed, encode_host, iterations=1000)
op10000 = sharded_pipeline(hostfeed, rb.outfeed, encode_host, iterations=10000)


"""


