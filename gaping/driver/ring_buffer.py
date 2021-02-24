from .. import driver as driver_lib
from .. import tf_tools as tft

import tensorflow as tf

from . import event_sequencer
from . import event_sink
from . import cursored
from . import sequence
from . import util
from ..util import EasyDict


class RingBufferFields:
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

  def fill(self, event_factory):
    # for (int i = 0; i < bufferSize; i++)
    # {
    #     entries[BUFFER_PAD + i] = eventFactory.newInstance();
    # }
    self.infeed, self.outfeed = event_factory(self.buffer_size)
  
  def element_at(self, feed, sequence):
    #return feed.peek(sequence)
    key, value = feed.get(sequence) # removes from feed
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

from gaping.driver import pipe; reload(pipe); builder = pipe.InfeedOutfeedBuilder()

builder.input('label', tf.int32); builder.input('image', tf.string); builder.output('label', tf.float32, shape=[1000]); builder.output('image', tf.float32, shape=[16,16,3]); 

from gaping.driver import ring_buffer; reload(ring_buffer); rb = ring_buffer.RingBuffer(lambda buffer_size: builder.get(), seqsr)

def rand_label(n=1000):
  return tf.random.uniform(minval=0, maxval=n, dtype=tf.int32, shape=())

def rand_jpeg(width=16, height=16):
  return tf.io.encode_jpeg(tf.cast(255*tf.random.uniform(shape=[width, height, 3]), tf.uint8))

inop = rb.iput(EasyDict(label=rand_label(), image=rand_jpeg()))

def parse_jpeg(e, seq=None):
  return EasyDict(
    label=tf.one_hot(e.label, 1000),
    image=tft.with_shape([16,16,3], lambda: tft.transform_image(e.image)))

pub = rb.try_publish_event(parse_jpeg)

with util.dep(inop):
  thru = rb.try_publish_event(parse_jpeg)

for i in tqdm.trange(BUFFER_SIZE): r( inop )

for i in tqdm.trange(BUFFER_SIZE): r( pub )

"""
