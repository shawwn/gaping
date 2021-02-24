from . import event_translator
from . import initializable
from . import ring_buffer
from . import pipe
from . import no_op_event_processor
from . import util

import tensorflow as tf

from contextlib import ExitStack


class LongEvent:
  @staticmethod
  def FACTORY(buffer_size=None):
    infeed_builder = pipe.PipeBuilder();
    outfeed_builder = pipe.PipeBuilder();
    infeed_builder.channel('value', tf.int32)
    outfeed_builder.channel('value', tf.int32)
    outfeed_builder.channel('index', tf.int64)
    infeed = infeed_builder.get()
    outfeed = outfeed_builder.get()
    if buffer_size is not None:
      def body(i):
        with util.dep(infeed.put(util.ti64(i), {'value': i})):
          return i + 1
      op = util.loop_reduce(body, buffer_size)
      op.eval()
    return infeed, outfeed


class EventPublisherTest(
    event_translator.EventTranslator,
    initializable.Initializable):

  BUFFER_SIZE = 32

  def __init__(self):
    self.ring_buffer = ring_buffer.RingBuffer.create_single_producer(LongEvent.FACTORY, EventPublisherTest.BUFFER_SIZE)

  def get_initializers(self):
    return super().get_initializers() + \
        self.ring_buffer.get_initializers()

  def should_publish_event(self):
    self.ring_buffer.add_gating_sequences(no_op_event_processor.NoOpEventProcessor(self.ring_buffer).get_sequence())

    with ExitStack() as stack:
      stack.enter_context(util.dep(self.ring_buffer.try_publish_event(self)))
      stack.enter_context(util.dep(self.ring_buffer.try_publish_event(self)))
      out = [self.ring_buffer.get()]
      stack.enter_context(util.dep(tf.group(out)))
      out += [self.ring_buffer.get()]
      return out

  def translate_to(self, event, sequence, *args, **kws):
    event.value = event.value + 29
    return event
    
