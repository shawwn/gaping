from typing_extensions import Protocol

from abc import abstractmethod

from . import exception_handler

import tensorflow as tf

import sys


class IgnoreExceptionHandler(exception_handler.ExceptionHandler):

  def __init__(self, logger=None):
    if logger is None:
      logger = tf.logging
    self.logger = logger

  def handle_event_exception(self, ex, sequence, event):
    self.logger.info("Exception processing: %s %s %s", sequence, event, ex)

  def handle_on_start_exception(self, ex):
    self.logger.info("Exception during onStart(): %s", ex)

  def handle_on_shutdown_exception(self, ex):
    self.logger.info("Exception during onShutdown(): %s", ex)






