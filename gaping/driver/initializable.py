from typing_extensions import Protocol

from abc import abstractmethod

import tensorflow as tf


class Initializable(Protocol):

  @abstractmethod
  def get_initializers(self):
    return list()

  @property
  def initializer(self):
    return tf.group(self.get_initializers())


