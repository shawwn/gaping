from typing_extensions import Protocol

from abc import abstractmethod

from . import exception_handler

import tensorflow as tf

import sys


# https://www.ianbicking.org/blog/2007/09/re-raising-exceptions.html
def reraise(ex):
  #raise ex
  exc_info = sys.exc_info()
  raise exc_info[0], exc_info[1], exc_info[2]


class FatalExceptionHandler(exception_handler.ExceptionHandler):

  def __init__(self, logger=None):
    if logger is None:
      logger = tf.logging
    self.logger = logger

  def handle_event_exception(self, ex, sequence, event):
    self.logger.fatal("Exception processing: %s %s %s", sequence, event, ex)
    reraise(ex)

  def handle_on_start_exception(self, ex):
    self.logger.fatal("Exception during onStart(): %s", ex)

  def handle_on_shutdown_exception(self, ex):
    self.logger.fatal("Exception during onShutdown(): %s", ex)





